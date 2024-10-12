import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Union, List


class WhisperHandler:
    def __init__(self, model_name="openai/whisper-large-v3"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    
    def transcribe(self, audio_input: Union[str, List[str]], batch_size: int = 1):
        """
        Transcribe one or multiple audio files.
        
        :param audio_input: Path to a single audio file or a list of paths to multiple audio files
        :param batch_size: Number of audio files to process in parallel (default: 1)
        :return: A dictionary or list of dictionaries containing the transcription result(s)
        """
        if isinstance(audio_input, str):
            result = self.pipe(audio_input)
        elif isinstance(audio_input, list):
            result = self.pipe(audio_input, batch_size=batch_size)
        else:
            raise ValueError("audio_input must be a string (file path) or a list of strings (file paths)")
        
        return result
