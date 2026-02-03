import os
import abc
from typing import Dict, Any, List
from loguru import logger

class VoiceEngineBase(abc.ABC):
    """
    Abstract Base Class for Voice Engines.
    """
    @abc.abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe an audio file and return text and metadata."""
        pass

class WhisperEngine(VoiceEngineBase):
    """
    ASR using OpenAI Whisper.
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size or "base"
        self.model = None

    def _lazy_load(self):
        if self.model is None:
            try:
                import whisper
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading Whisper model: {self.model_size} on {device}...")
                self.model = whisper.load_model(self.model_size, device=device)
                logger.debug("Whisper model loaded successfully")
            except ImportError:
                logger.error("openai-whisper not installed. Please install with 'pip install openai-whisper'")
                raise
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        self._lazy_load()
        logger.info(f"Using Whisper ({self.model_size}) to transcribe: {audio_path}")
        
        result = self.model.transcribe(audio_path)
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language"),
            "engine": f"whisper-{self.model_size}",
            "metadata": {
                "model_size": self.model_size,
                "file": audio_path
            }
        }

class VoiceEngineFactory:
    """
    Factory to create Voice engines.
    """
    @staticmethod
    def get_engine(engine_type: str, **kwargs) -> VoiceEngineBase:
        if engine_type == "whisper":
            return WhisperEngine(model_size=kwargs.get("model_size"))
        else:
            raise ValueError(f"Unknown voice engine type: {engine_type}")
