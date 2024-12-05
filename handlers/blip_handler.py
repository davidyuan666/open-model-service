import torch
from PIL import Image
import os
from pathlib import Path
from models.blip import blip_decoder
from utils.blip_util import BlipUtil
from utils.cos_util import COSUtil
from urllib.parse import urlparse

class BlipHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_util = BlipUtil()
        self.cos_util = COSUtil()
        self.init_model()

    '''
    https://github.com/salesforce/LAVIS/tree/main/projects/blip2

    featurize port export 5000
    https://docs.featurize.cn/docs/manual/port-exporting  (端口转发，最多支持10个)

    workspace.featurize.cn:44768

    '''
    def init_model(self, gpu_id=0):
        try:
            # Set device
            self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            print(f"Device: {self.device}")

            # Create checkpoints directory if it doesn't exist
            if not Path("checkpoints").is_dir():
                print("Checkpoint directory not found.")
                try:
                    self.blip_util.create_dir("checkpoints")
                except Exception as e:
                    raise RuntimeError(f"Failed to create checkpoint directory: {str(e)}")

            # Download checkpoint if it doesn't exist
            checkpoint_path = Path("checkpoints/model_large_caption.pth")
            if not checkpoint_path.is_file():
                try:
                    print("Downloading checkpoint...")
                    self.blip_util.download_checkpoint()
                except Exception as e:
                    raise RuntimeError(f"Failed to download checkpoint: {str(e)}")

            # Verify checkpoint file exists and has content
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                raise RuntimeError("Checkpoint file is missing or empty")

            # Load model with retry mechanism
            print("Checkpoint loading...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print('checkpoint_path', checkpoint_path)
                    self.model = blip_decoder(
                        pretrained=checkpoint_path,
                        image_size=384,
                        vit="large",
                        med_config=os.path.join("configs", "med_config.json")
                    )
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    print(f"Model loaded successfully to {self.device}")
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying...")
                        continue
                    raise RuntimeError(f"Failed to initialize model after {max_retries} attempts: {str(e)}")

        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def download_image_from_cos(self, cos_image_url, project_no):
        """
        Download a file from COS to local storage
        
        Args:
            cos_video_url (str): COS URL of the file
            project_no (str): Project number for organizing local storage
            
        Returns:
            str: Local path of downloaded file or None if download fails
        """
        try:
            parsed_url = urlparse(cos_image_url)
            if not parsed_url.path:
                raise ValueError("Invalid COS URL: no path found")

            object_key = parsed_url.path.lstrip('/')
            original_filename = os.path.basename(object_key)
            
            if not original_filename:
                raise ValueError("Invalid COS URL: no filename found")
            
            image_local_path = os.path.join(os.getcwd(), 'temp', project_no, original_filename)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_local_path), exist_ok=True)

            # Download the file
            self.cos_util.download_file(self.cos_util.bucket_name, object_key, image_local_path)
            
            if not os.path.exists(image_local_path):
                raise FileNotFoundError("File download failed: file not found at destination")

            return image_local_path
            
        except Exception as e:
            print(f"Error getting file from COS URL {cos_image_url}: {str(e)}")
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
            local_path = self.download_image_from_cos(image_url, project_no)
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
            transformed_images = self.blip_util.prep_images([image])
            if not transformed_images:
                raise ValueError("Failed to transform image")

            transformed_image = transformed_images[0]
            if transformed_image.device != self.device:
                transformed_image = transformed_image.to(self.device)



            # 生成描述
            with torch.no_grad():
                caption = self.model.generate(
                    transformed_image,  # Take first (and only) transformed image
                    sample=False, 
                    num_beams=3, 
                    max_length=20, 
                    min_length=5
                )
                if caption is not None:
                    caption_content = caption[0]
                    return caption_content  # Return first (and only) caption
                return None
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return None


