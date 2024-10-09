import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class StableDiffusionHandler:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(os.getcwd(), 'models', model_id)
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
        with torch.no_grad():
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, 'temp.png')
            image.save(image_path)
        
        return image

    def save_image(self, image, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)