import unittest
import os
from handlers.whisper_handler import WhisperHandler

class TestWhisperHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.whisper_handler = WhisperHandler(model_name="base")
        # 确保这些音频文件存在于指定路径
        cls.english_audio_path = "tests/test_audio/english_sample.mp3"
        cls.chinese_audio_path = "tests/test_audio/chinese_sample.mp3"

    def test_transcribe_english(self):
        result = self.whisper_handler.transcribe(self.english_audio_path)
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertGreater(len(result["text"]), 0)

    def test_transcribe_chinese(self):
        result = self.whisper_handler.transcribe(self.chinese_audio_path)
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertGreater(len(result["text"]), 0)

    def test_detect_language_english(self):
        language = self.whisper_handler.detect_language(self.english_audio_path)
        self.assertEqual(language, "en")

    def test_detect_language_chinese(self):
        language = self.whisper_handler.detect_language(self.chinese_audio_path)
        self.assertEqual(language, "zh")

    def test_translate_chinese_to_english(self):
        result = self.whisper_handler.translate(self.chinese_audio_path, target_language="en")
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertGreater(len(result["text"]), 0)
        # 这里我们可以添加一些基本的检查来确保输出是英语
        self.assertTrue(any(word in result["text"].lower() for word in ["the", "a", "an", "is", "are"]))

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.whisper_handler.transcribe("non_existent_file.mp3")

if __name__ == '__main__':
    unittest.main()