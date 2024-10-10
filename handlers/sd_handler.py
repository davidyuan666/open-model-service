import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from modelscope import snapshot_download

class StableDiffusionHandler:
    def __init__(self, model_id="stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.base_path = os.path.join(os.getcwd(), 'models')
        self.model_path = os.path.join(self.base_path, 'AI-ModelScope',self.model_id)
        
        os.makedirs(self.model_path, exist_ok=True)

        if not any(os.scandir(self.model_path)):
            print(f"Local model not found. Downloading from ModelScope: {self.model_id}")
            self.download_from_modelscope()
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)
        except (ValueError, ImportError, OSError) as e:
            if "protobuf" in str(e):
                print("Error: Protobuf library not found. Please install it using:")
                print("pip install protobuf")
                print("You may need to restart your runtime after installation.")
                raise SystemExit(1)
            elif "model_index.json" in str(e):
                print(f"Local model files are not in the correct format. Downloading from Hugging Face: {self.model_id}")
            else:
                print(f"Failed to load local model. Downloading from Hugging Face: {self.model_id}")
            
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(f"stable-diffusion-v1-5/{self.model_id}", torch_dtype=torch.float16)
                # Save the correctly formatted model locally
                self.pipe.save_pretrained(self.model_path)
                print(f"Model saved to: {self.model_path}")
            except Exception as e:
                print(f"Failed to load model from Hugging Face: {str(e)}")
                raise SystemExit(1)
        
        self.pipe = self.pipe.to(self.device)

    def download_from_modelscope(self):
        try:
            snapshot_download(
                f'AI-ModelScope/{self.model_id}',
                cache_dir=self.base_path
            )
            print(f"Model downloaded to: {self.model_path}")
        except ImportError:
            print("ModelScope not installed. Please install it with 'pip install modelscope'")
        except Exception as e:
            print(f"Failed to download model from ModelScope: {str(e)}")

    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
        with torch.no_grad():
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]        
        return image

    def save_image(self, image, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

def main():
    handler = StableDiffusionHandler()
    prompt = "A beautiful sunset over a calm ocean"
    image = handler.generate_image(prompt)
    output_path = os.path.join(os.getcwd(), 'output', 'generated_image.png')
    handler.save_image(image, output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()