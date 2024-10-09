import unittest
import os
from PIL import Image
from handlers.sd_handler import StableDiffusionHandler

class TestStableDiffusionHandler(unittest.TestCase):
    def setUp(self):
        self.handler = StableDiffusionHandler()
        self.test_prompt = "A beautiful sunset over the ocean"
        self.test_output_path = os.path.join(os.getcwd(), "tests", "output", "test_image.png")
        self.temp_image_path = os.path.join(os.getcwd(), "temp", "temp.png")

    def test_generate_image(self):
        image = self.handler.generate_image(self.test_prompt)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))  # 默认图像大小
        self.assertTrue(os.path.exists(self.temp_image_path))

    def test_save_image(self):
        image = self.handler.generate_image(self.test_prompt)
        self.handler.save_image(image, self.test_output_path)
        self.assertTrue(os.path.exists(self.test_output_path))
        saved_image = Image.open(self.test_output_path)
        self.assertEqual(saved_image.size, image.size)

    def tearDown(self):
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
        if os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)

if __name__ == '__main__':
    unittest.main()