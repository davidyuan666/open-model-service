import whisper
import torch

class WhisperHandler:
    def __init__(self, model_name="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)

    def transcribe(self, audio_path):
        """
        Transcribe the given audio file.
        
        :param audio_path: Path to the audio file
        :return: A dictionary containing the transcription result
        """
        result = self.model.transcribe(audio_path)
        return result

    def detect_language(self, audio_path):
        """
        Detect the language of the given audio file.
        
        :param audio_path: Path to the audio file
        :return: Detected language code
        """
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)
        
        return max(probs, key=probs.get)

    def translate(self, audio_path, target_language="en"):
        """
        Translate the audio content to the target language.
        
        :param audio_path: Path to the audio file
        :param target_language: Target language code (default is English)
        :return: A dictionary containing the translation result
        """
        result = self.model.transcribe(audio_path, task="translate", language=target_language)
        return result
