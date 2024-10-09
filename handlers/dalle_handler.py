import os
import requests
from PIL import Image
from io import BytesIO
import openai

class DALLEHandler:
    def __init__(self):
        # 设置 OpenAI API 密钥
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

    def generate_image(self, prompt, n=1, size="1024x1024"):
        """
        使用 DALL-E 模型生成图像
        
        :param prompt: 描述所需图像的文本提示
        :param n: 要生成的图像数量（默认为1）
        :param size: 图像尺寸，可选 "256x256", "512x512", 或 "1024x1024"（默认）
        :return: 生成图像的URL列表
        """
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=n,
                size=size
            )
            return [item['url'] for item in response['data']]
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            return []

    def save_image(self, image_url, save_path):
        """
        下载并保存图像
        
        :param image_url: 图像的URL
        :param save_path: 保存图像的路径
        """
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")
