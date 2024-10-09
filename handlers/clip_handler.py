import torch
import clip
from PIL import Image

class CLIPHandler:
    '''
    RN50
    '''
    def __init__(self, image_model_name="ViT-B/32", text_model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_model, self.image_preprocess = clip.load(image_model_name, device=self.device)
        self.text_model, _ = clip.load(text_model_name, device=self.device)

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.image_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.image_model.encode_image(image_input)
        return image_features

    def encode_text(self, text):
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.text_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化嵌入向量
        return text_features

    def calculate_similarity(self, image_features, text_features):
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        return similarity.item()

    def classify_image(self, image_path, categories):
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(categories)
        
        similarities = (image_features @ text_features.T).squeeze(0)
        best_match = categories[similarities.argmax().item()]
        
        return best_match, similarities.tolist()