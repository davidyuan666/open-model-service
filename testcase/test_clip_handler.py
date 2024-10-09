import unittest
import torch
from handlers.clip_handler import CLIPHandler
import os

class TestCLIPHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip_handler_vit = CLIPHandler(image_model_name="ViT-B/32")
        cls.clip_handler_resnet = CLIPHandler(image_model_name="RN50")
        cls.test_image_path = "https://plus.unsplash.com/premium_photo-1681746821577-5c5f32433be5?q=80&w=1966&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

    def test_encode_image_vit(self):
        image_features = self.clip_handler_vit.encode_image(self.test_image_path)
        self.assertIsInstance(image_features, torch.Tensor)
        self.assertEqual(image_features.shape[1], 512)  # ViT-B/32 模型的特征维度是 512

    def test_encode_image_resnet(self):
        image_features = self.clip_handler_resnet.encode_image(self.test_image_path)
        self.assertIsInstance(image_features, torch.Tensor)
        self.assertEqual(image_features.shape[1], 1024)  # RN50 模型的特征维度是 1024

    def test_encode_text(self):
        text = "这是一个测试文本"
        for handler in [self.clip_handler_vit, self.clip_handler_resnet]:
            text_features = handler.encode_text(text)
            self.assertIsInstance(text_features, torch.Tensor)
            self.assertEqual(text_features.shape[1], 512)  # 文本特征维度总是 512

    def test_calculate_similarity(self):
        for handler in [self.clip_handler_vit, self.clip_handler_resnet]:
            image_features = handler.encode_image(self.test_image_path)
            text_features = handler.encode_text("一只猫")
            similarity = handler.calculate_similarity(image_features, text_features)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, -1)
            self.assertLessEqual(similarity, 1)

    def test_classify_image(self):
        categories = ["猫", "狗", "鸟"]
        for handler in [self.clip_handler_vit, self.clip_handler_resnet]:
            best_match, similarities = handler.classify_image(self.test_image_path, categories)
            self.assertIn(best_match, categories)
            self.assertEqual(len(similarities), len(categories))
            for similarity in similarities:
                self.assertGreaterEqual(similarity, -1)
                self.assertLessEqual(similarity, 1)

    def test_get_image_model_type(self):
        self.assertEqual(self.clip_handler_vit.get_image_model_type(), "ViT")
        self.assertEqual(self.clip_handler_resnet.get_image_model_type(), "ResNet")

if __name__ == '__main__':
    unittest.main()