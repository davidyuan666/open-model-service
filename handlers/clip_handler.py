import torch
import clip
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sentence_transformers import SentenceTransformer, util
import os

class CLIPHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eng_model = None
        self.chn_model = None
        self.chn_processor = None
        self.cache_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.cache_dir, exist_ok=True)

    def init_eng_clip(self, model_name="ViT-B/32"):
        model_path = os.path.join(self.cache_dir, f'clip-{model_name}')
        if os.path.exists(model_path):
            self.eng_model, self.eng_preprocess = clip.load(model_path, device=self.device)
        else:
            self.eng_model, self.eng_preprocess = clip.load(model_name, device=self.device)
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.eng_model.state_dict(), os.path.join(model_path, 'model.pt'))
        self.eng_model.eval()

    def init_chn_clip(self):
        model_path = os.path.join(self.cache_dir, 'chinese-clip')
        if os.path.exists(model_path):
            self.chn_model = ChineseCLIPModel.from_pretrained(model_path).to(self.device)
            self.chn_processor = ChineseCLIPProcessor.from_pretrained(model_path)
        else:
            self.chn_model = ChineseCLIPModel.from_pretrained("TencentARC/QA-CLIP-ViT-B-16", cache_dir=model_path).to(self.device)
            self.chn_processor = ChineseCLIPProcessor.from_pretrained("TencentARC/QA-CLIP-ViT-B-16", cache_dir=model_path)
        self.chn_model.eval()

    def encode_image_eng(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.eng_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.eng_model.encode_image(image_input)
        return image_features

    def encode_text_eng(self, text):
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.eng_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image_chn(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.chn_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.chn_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def encode_text_chn(self, text):
        inputs = self.chn_processor(text=text, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.chn_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def calculate_similarity(self, image_features, text_features):
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        return similarity.item()

    def classify_image(self, image_path, categories, language='eng'):
        if language == 'eng':
            if self.eng_model is None:
                self.init_eng_clip()
            image_features = self.encode_image_eng(image_path)
            text_features = self.encode_text_eng(categories)
        elif language == 'chn':
            if self.chn_model is None:
                self.init_chn_clip()
            image_features = self.encode_image_chn(image_path)
            text_features = self.encode_text_chn(categories)
        else:
            raise ValueError("Unsupported language. Choose 'eng' or 'chn'.")
        
        similarities = (image_features @ text_features.T).squeeze(0)
        best_match = categories[similarities.argmax().item()]
        
        return best_match, similarities.tolist()
    


def main():
    handler = CLIPHandler()

    image_path = os.path.join(os.getcwd(), 'output', 'generated_image.png')

    # English classification
    # eng_categories = ['dog', 'cat', 'bird', 'fish', 'horse']
    # best_match_eng, similarities_eng = handler.classify_image(image_path, eng_categories, language='eng')
    # print(f"English classification result: {best_match_eng}")
    # print(f"English similarities: {similarities_eng}")

    # Chinese classification
    chn_categories = ['狗', '猫', '鸟', '鱼', '马']
    best_match_chn, similarities_chn = handler.classify_image(image_path, chn_categories, language='chn')
    print(f"Chinese classification result: {best_match_chn}")
    print(f"Chinese similarities: {similarities_chn}")

if __name__ == "__main__":
    main()

