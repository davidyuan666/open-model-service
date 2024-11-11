import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
from pathlib import Path
from models.blip import blip_decoder
from utils.blip_util import BlipUtil

class BlipHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化 BLIP-2 模型
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.blip_util = BlipUtil()

    def generate_description(self, image_path, max_length=50, num_beams=5):
        """
        生成图像描述
        :param image_path: 图像路径
        :param max_length: 生成描述的最大长度
        :param num_beams: beam search的束宽
        :return: 生成的描述文本
        """
        try:
            # 加载并处理图像
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False
                )
                
            # 解码生成的文本
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def generate_description_with_prompt(self, image_path, prompt, max_length=50):
        """
        使用特定提示生成图像描述
        :param image_path: 图像路径
        :param prompt: 提示文本
        :param max_length: 生成描述的最大长度
        :return: 生成的描述文本
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=False
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error generating description with prompt: {e}")
            return None

    '''
    https://github.com/salesforce/LAVIS/tree/main/projects/blip2

    featurize port export 5000
    https://docs.featurize.cn/docs/manual/port-exporting  (端口转发，最多支持10个)

    workspace.featurize.cn:44768

    '''
    def init_model(self,gpu_id=0):
        """初始化模型"""
        global model, device
        
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # 检查并创建checkpoints目录
        if not Path("checkpoints").is_dir():
            print("checkpoint directory not found.")
            utils.create_dir("checkpoints")

        # 下载模型检查点
        if not Path("checkpoints/model_large_caption.pth").is_file():
            utils.download_checkpoint()

        print("Checkpoint loading...")
        model = blip_decoder(
            pretrained="./checkpoints/model_large_caption.pth", 
            image_size=384, 
            vit="large"
        )
        model.eval()
        model = model.to(device)
        print(f"Model loaded to {device}")

    def download_video_from_cos(self,cos_video_url, project_no):
            try:
                cos_util = COSOperationsUtil()
                
                parsed_url = urlparse(cos_video_url)

                object_key = parsed_url.path.lstrip('/')
                
                # Extract the original filename from the object_key
                original_filename = os.path.basename(object_key)
                
                video_local_path = os.path.join(os.getcwd(), 'temp', project_no, original_filename)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(video_local_path), exist_ok=True)

                cos_util.download_file(cos_util.bucket_name, object_key, video_local_path)


                return video_local_path
            
            except Exception as e:
                print(f"Error getting video from COS URL {cos_video_url}: {str(e)}")
                return None
        

    def load_image_from_cos(self,image_url, project_no):
        """从COS加载图片
        
        Args:
            image_url (str): COS中图片的URL
            project_no (str): 项目编号
            
        Returns:
            PIL.Image: 加载的图片对象
            
        Raises:
            Exception: 当图片加载失败时抛出异常
        """
        try:
            # 从COS下载文件到本地
            local_path = self.download_video_from_cos(image_url, project_no)
            if not local_path:
                raise Exception("Failed to download file from COS")
                
            # 打开并转换图片
            try:
                image = Image.open(local_path).convert('RGB')
                return image
            except Exception as e:
                raise Exception(f"Error opening image file: {str(e)}")
            finally:
                # 可选：清理临时文件
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {local_path}: {str(e)}")
                    
        except Exception as e:
            raise Exception(f"Error loading image from COS URL {image_url}: {str(e)}")



    def generate_caption(self, image_url, project_no):
        """
        Generate caption for an image from COS URL
        
        Args:
            image_url (str): COS URL of the image
            project_no (str): Project number
            
        Returns:
            str: Generated caption or None if error occurs
        """
        try:
            # 加载图片
            image = self.load_image_from_cos(image_url, project_no)
            if image is None:
                raise ValueError("Failed to load image from COS")
            
            # 预处理图片
            transformed_image = self.blip_util.prep_images([image])
            if not transformed_image:
                raise ValueError("Failed to transform image")

            # 生成描述
            with torch.no_grad():
                caption = self.model.generate(
                    transformed_image[0],  # Take first (and only) transformed image
                    sample=False, 
                    num_beams=3, 
                    max_length=20, 
                    min_length=5
                )

                print(f'caption: {caption}')
                # BLIP model returns the caption directly as text
                return caption[0]  # Return first (and only) caption
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return None


