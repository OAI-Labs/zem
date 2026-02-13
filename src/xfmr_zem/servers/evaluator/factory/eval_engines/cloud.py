from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests
import os
from dataclasses import dataclass, field
from enum import Enum
import json

# --- Giả lập OpikBaseModel (Giữ nguyên) ---
try:
    from opik.evaluation.models import OpikBaseModel
except ImportError:
    class OpikBaseModel(ABC):
        def __init__(self, model_name: str):
            self.model_name = model_name
        
        @abstractmethod
        def generate_string(self, input: str, response_format: Any = None, **kwargs: Any) -> str:
            pass

        @abstractmethod
        def generate_provider_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
            pass

# --- 1. Cấu hình cơ bản (Giữ nguyên) ---

class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model_name: str
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})

# --- 2. Class Cha (ĐÃ SỬA LỖI) ---

class OpikCloudModel(OpikBaseModel, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__(model_name=config.model_name)
        self.config = config
        self.headers = config.headers
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def extract_content(self, response: Dict[str, Any]) -> str:
        pass

    def generate_string(self, input: str, response_format: Any = None, **kwargs) -> str:
        # Lưu ý: response_format thường là Pydantic Class.
        # Ta không truyền nó vào kwargs của provider trừ khi đã convert sang JSON Schema.
        messages = [{"role": "user", "content": input}]
        
        # Gọi hàm provider, truyền kwargs (nhưng không truyền response_format dạng class)
        response = self.generate_provider_response(messages=messages, **kwargs)
        return self.extract_content(response)
    
    def generate_provider_response(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        payload = self.format_messages(messages)
        
        # --- FIX: Lọc kwargs để tránh lỗi JSON Serializable ---
        safe_kwargs = kwargs.copy()
        
        # Kiểm tra và loại bỏ 'response_format' nếu nó không phải là dict (tức là nó là Class Pydantic)
        # API Cloud không hiểu Python Class, nó chỉ hiểu JSON Dict.
        if "response_format" in safe_kwargs:
            if not isinstance(safe_kwargs["response_format"], dict):
                # Xóa bỏ để tránh crash. 
                # (Nếu muốn hỗ trợ structured output, bạn phải code thêm logic convert Pydantic -> JSON Schema ở đây)
                del safe_kwargs["response_format"]

        # Cập nhật payload với các tham số an toàn
        payload.update(safe_kwargs)
        
        try:
            # Debug: In ra payload để kiểm tra nếu còn lỗi
            # print(f"Sending payload to {self.config.model_name}: {payload.keys()}")
            
            response = requests.post(
                self.config.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request failed: {e}"
            if 'response' in locals():
                error_msg += f" | Body: {response.text}"
            raise RuntimeError(error_msg)
        except TypeError as e:
            # Bắt lỗi JSON serialization cụ thể hơn để dễ debug
            raise TypeError(f"JSON Serialization Error: {e}. Check if kwargs contains Python Classes.")

# --- 3. Các Model cụ thể (Giữ nguyên) ---

class OpikOpenAIModel(OpikCloudModel):
    def format_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": messages
        }
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

class OpikGeminiModel(OpikCloudModel):
    def format_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        contents = []
        for msg in messages:
            contents.append({
                "parts": [{"text": msg["content"]}],
                "role": "user" if msg["role"] == "user" else "model"
            })
        return {
            "contents": contents,
            "generationConfig": {"temperature": 0.7}
        }
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""

class OpikAnthropicModel(OpikCloudModel):
    def format_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": 1000
        }
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        return response["content"][0]["text"]

# --- 4. Factory (Giữ nguyên) ---

class CloudModelFactory:
    _REGISTRY = {
        Provider.OPENAI: (OpikOpenAIModel, "OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions"),
        Provider.ANTHROPIC: (OpikAnthropicModel, "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages"),
        Provider.GEMINI: (OpikGeminiModel, "GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"),
    }

    @classmethod
    def create_model(cls, provider: str, model_id: str, api_key: str = None, base_url: str = None, **kwargs) -> OpikCloudModel:
        provider = provider.lower()
        if provider not in cls._REGISTRY:
            available = ", ".join([p.value for p in Provider])
            raise ValueError(f"Unsupported provider: {provider}. Available: {available}")

        model_cls, env_var, default_url_template = cls._REGISTRY[provider]

        if not api_key:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"API Key not found for {provider} (checked env: {env_var})")

        if not base_url:
            base_url = default_url_template.replace("{model}", model_id)
        
        headers = {"Content-Type": "application/json"}
        if provider == Provider.ANTHROPIC:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider == Provider.GEMINI:
            headers["x-goog-api-key"] = api_key 
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        config = ModelConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model_id,
            headers=headers
        )
        
        return model_cls(config)