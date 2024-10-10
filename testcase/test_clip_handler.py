import unittest
import os
import sys
import torch

# Add the parent directory to the Python path to import CLIPHandler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handlers.clip_handler import CLIPHandler

class TestCLIPHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = CLIPHandler()
        cls.test_image_path = os.path.join(os.getcwd(), 'tests', 'test_image.jpg')
        
        # Ensure test image exists
        if not os.path.exists(cls.test_image_path):
            raise FileNotFoundError(f"Test image not found at {cls.test_image_path}")

    def test_init(self):
        self.assertIsNotNone(self.handler)
        self.assertTrue(os.path.exists(self.handler.cache_dir))

    def test_eng_clip(self):
        categories = ['dog', 'cat', 'bird', 'fish', 'horse']
        best_match, similarities = self.handler.classify_image(self.test_image_path, categories, language='eng')
        
        self.assertIn(best_match, categories)
        self.assertEqual(len(similarities), len(categories))
        self.assertTrue(all(isinstance(sim, float) for sim in similarities))

    def test_chn_clip(self):
        categories = ['狗', '猫', '鸟', '鱼', '马']
        best_match, similarities = self.handler.classify_image(self.test_image_path, categories, language='chn')
        
        self.assertIn(best_match, categories)
        self.assertEqual(len(similarities), len(categories))
        self.assertTrue(all(isinstance(sim, float) for sim in similarities))

    def test_invalid_language(self):
        with self.assertRaises(ValueError):
            self.handler.classify_image(self.test_image_path, ['test'], language='invalid')

    def test_model_caching(self):
        # Test if models are cached properly
        self.handler.init_eng_clip()
        self.handler.init_chn_clip()
        
        eng_model_path = os.path.join(self.handler.cache_dir, 'clip-ViT-B/32')
        chn_model_path = os.path.join(self.handler.cache_dir, 'chinese-clip')
        
        self.assertTrue(os.path.exists(eng_model_path))
        self.assertTrue(os.path.exists(chn_model_path))

if __name__ == '__main__':
    unittest.main()