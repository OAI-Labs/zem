from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests
import os
from dataclasses import dataclass
from enum import Enum
from .eval_engines.cloud import CloudModelFactory
from .eval_engines.local import OpikLocalFactory
from loguru import logger

class EvaluateModelFactory:
    @staticmethod
    def get_model(
        engine: str, 
        model_id: Optional[str] = None, 
        provider: Optional[str] = None, 
        **kwargs
    ) -> Any:
        """
        Tạo model đánh giá dựa trên engine.
        Nếu engine='cloud', bắt buộc phải có provider và model_id.
        """
        
        # 1. Xử lý Engine Cloud
        if engine == "cloud":
            if not provider or not model_id:
                raise ValueError("Engine 'cloud' yêu cầu phải có 'provider' và 'model_id'.")
            
            # Gọi lại Factory cũ để build model
            return CloudModelFactory.create_model(
                provider=provider,
                model_id=model_id,
                **kwargs
            )
            
        # 2. Xử lý các Engine khác (ví dụ sau này)
        elif engine == "local":
            logger.info(model_id, provider)
            if not model_id or not provider:
                raise ValueError("Engine 'local' yêu cầu phải có 'model_id' và 'provider'")
            return OpikLocalFactory.create_model(provider=provider, model_id=model_id, **kwargs)
            
        # 3. Lỗi nếu không tìm thấy engine
        else:
            raise ValueError(f"Engine không hợp lệ: {engine}")

# ==========================================
# Cách sử dụng
# ==========================================
if __name__ == "__main__":
    # Case 1: Cloud - OpenAI
    gpt = EvaluateModelFactory.create(
        engine="cloud",
        provider="openai",
        model_id="gpt-4",
        api_key="sk-..." 
    )
    
    # Case 2: Cloud - Azure (truyền thêm tham số qua kwargs)
    azure = EvaluateModelFactory.create(
        engine="cloud",
        provider="azure",
        model_id="gpt-35",
        base_url="https://...",
        api_version="2023-05-15"
    )