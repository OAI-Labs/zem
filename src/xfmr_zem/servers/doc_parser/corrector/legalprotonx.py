"""
File: /app/corrector/protonx.py
Description: Wrapper class cho model sửa lỗi chính tả (Vietnamese Correction).
"""

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

from .corrector import register
from loguru import logger

@register("legalprotonx")
class VietnameseLegalCorrector:
    """
    Class xử lý sửa lỗi chính tả tiếng Việt sử dụng model Seq2Seq.
    """

    def __init__(self, model_path: str, tokenizer: dict, seq2seq: dict, config_path: Optional[str] = None):
        """
        Khởi tạo VietnameseCorrector.

        Args:
            model_path (str): Đường dẫn model hoặc tên model trên HuggingFace.
            config_path (str, optional): Đường dẫn file config .yaml.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Loading VietnameseCorrector model from: {model_path} ({self.device})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       device_map = "auto")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                           device_map = "auto")
        self.tokenizer_args = tokenizer
        self.seq2seq_args = seq2seq

    def _tokenize(self, text: str):
        """Tokenize input text."""
        tokenizer_kwargs = self.tokenizer_args.copy()
        # Đảm bảo return_tensors là 'pt' cho PyTorch
        tokenizer_kwargs['return_tensors'] = 'pt'
        
        inputs = self.tokenizer(text,
                                truncation=tokenizer_kwargs.get('truncation', True),
                                max_length=tokenizer_kwargs.get('max_length', 512),
                                return_tensors=tokenizer_kwargs.get('return_tensors', 'pt'))
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _generate(self, inputs):
        """Generate corrected sequences."""
        gen_kwargs = self.seq2seq_args.copy()
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, 
                                          num_beams=gen_kwargs.get('num_beams', 5),
                                          max_new_tokens=gen_kwargs.get('max_new_tokens', 512),
                                          early_stopping=gen_kwargs.get('early_stopping', True),
                                          return_dict_in_generate=gen_kwargs.get('return_dict_in_generate', True))
        return outputs

    def _decode(self, outputs):
        """Decode generated sequences to text."""
        # Xử lý output (nếu return_dict_in_generate=True thì kết quả nằm trong .sequences)
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs

        # 3. Decode
        corrected_text = self.tokenizer.batch_decode(sequences, 
                                                     skip_special_tokens=True)
        return corrected_text[0]

    def correct(self, text: str) -> str:
        """
        Thực hiện sửa lỗi cho văn bản đầu vào.

        Args:
            text (str): Văn bản cần sửa.

        Returns:
            str: Văn bản đã được sửa.
        """
        if not text or not text.strip():
            return text

        # 1. Tokenize
        inputs = self._tokenize(text)
        # 2. Generate
        outputs = self._generate(inputs)
        # 3. Decode
        return self._decode(outputs)
