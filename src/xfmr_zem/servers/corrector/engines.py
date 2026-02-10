import abc
from typing import Optional
from loguru import logger

class CorrectorEngineBase(abc.ABC):
    """
    Abstract Base Class for LM Correctors.
    """
    @abc.abstractmethod
    def correct(self, text: str) -> str:
        """Correct the given text."""
        pass

class HuggingFaceCorrectorEngine(CorrectorEngineBase):
    """
    Corrector using HuggingFace Seq2Seq models.
    Simplified: Uses device_map="auto" and internal imports.
    """
    def __init__(
        self, 
        model_id: str,
    ):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        
        # Load model immediately
        self._load_model()

    def _load_model(self):
        if not self.model_id:
            logger.warning("No model_id provided for HuggingFaceLM.")
            return

        logger.info(f"🚀 Loading HuggingFaceLM from: {self.model_id} with device_map='auto'")
        
        try:
            # Import transformers strictly inside this method
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Load tokenizer & model with device_map="auto"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                device_map="auto"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, 
                device_map="auto"
            )
            
        except ImportError:
            logger.error("❌ 'transformers', 'torch' or 'accelerate' library not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def correct(self, text: str) -> str:
        # Import torch inside method
        import torch

        if self.model is None or not text or not isinstance(text, str) or not text.strip():
            return text

        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Khi dùng device_map="auto", ta chuyển input vào device của model
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    num_beams=10,
                    max_new_tokens=512,
                    early_stopping=True
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during correction: {e}")
            return text

class CorrectorEngineFactory:
    """
    Factory class to create instances of Corrector Engines.
    """
    @staticmethod
    def get_engine(engine_type: str, **kwargs) -> CorrectorEngineBase:
        """
        Factory method to return the appropriate corrector engine.
        kwargs will be passed to the engine constructor if needed.
        """
        engine_type = engine_type.lower().strip()
        
        if engine_type == "huggingface":
            return HuggingFaceCorrectorEngine(model_id=kwargs.get("model_id"))
            
        else:
            raise ValueError(f"Unknown engine type for corrector: {engine_type}")