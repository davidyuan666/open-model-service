import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

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


