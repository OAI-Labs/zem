import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from opik.evaluation.models import OpikBaseModel

import os
import json
import requests
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
from opik.evaluation.models import OpikBaseModel

class OpikGeminiModel(OpikBaseModel):
    """
    Opik model wrapper for Google Gemini API.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model_name=model_name)
        
        # Load API key
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Default config
        self.default_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p,
            "topK": top_k
        }
        # Remove None values
        self.default_config = {k: v for k, v in self.default_config.items() if v is not None}

    def generate_string(
        self, 
        input: str, 
        response_format: Optional[Type[BaseModel]] = None, 
        **kwargs: Any
    ) -> str:
        """
        Simplified interface to generate a string output from the model.
        Matches OpikBaseModel signature strictly.
        """
        # Construct a simple user message
        messages = [{"role": "user", "content": input}]
        
        # Call the provider response
        response = self.generate_provider_response(
            messages=messages, 
            response_format=response_format, 
            **kwargs
        )
        
        # Extract text from Gemini response structure
        try:
            if (
                response.get("candidates") 
                and len(response["candidates"]) > 0
                and response["candidates"][0].get("content")
                and response["candidates"][0]["content"].get("parts")
            ):
                return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            pass
            
        raise ValueError(f"Unexpected response format from Gemini API: {response}")

    def generate_provider_response(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> Any:
        """
        Generate a provider-specific response.
        Matches OpikBaseModel signature strictly.
        """
        # 1. Separate System Prompt (if any) from Contents
        gemini_contents = []
        system_instruction = None

        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Handle list of dicts (multimodal) or string
            text_content = ""
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                # Simple extraction for text-only parts logic
                # You might need more complex logic here for images
                text_content = " ".join([item.get("text", "") for item in content if "text" in item])
            else:
                text_content = str(content)

            if role == "system":
                # Gemini handles system prompt via systemInstruction field
                system_instruction = {"parts": [{"text": text_content}]}
            else:
                # Map 'assistant' to 'model' for Gemini
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": text_content}]
                })

        # 2. Prepare Request Body
        request_body = {
            "contents": gemini_contents
        }
        if system_instruction:
            request_body["system_instruction"] = system_instruction

        # 3. Handle Configuration (Merge defaults with kwargs)
        # kwargs in generate call override init defaults
        generation_config = self.default_config.copy()
        
        # Map kwargs to Gemini API fields if they exist
        if "temperature" in kwargs: generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs: generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs: generation_config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs: generation_config["topK"] = kwargs["top_k"]

        # Handle response_format (Basic JSON mode support)
        # Note: Full schema validation requires more complex conversion from Pydantic to JSON Schema
        response_format = kwargs.get("response_format")
        if response_format:
             generation_config["responseMimeType"] = "application/json"

        if generation_config:
            request_body["generationConfig"] = generation_config

        # 4. Make API Request
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=request_body,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Gemini API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nResponse: {e.response.text}"
            raise RuntimeError(error_msg)

    # Optional: Implement async methods if you want full compliance, 
    # but OpikBaseModel abstract methods are the critical ones.

class EvaluateModelFactory:
    @staticmethod
    def get_model(engine: str, model_name: str, **kwargs) -> OpikBaseModel:
        if engine.lower() == "gemini":
            return OpikGeminiModel(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unknown evaluation model engine: {engine}")
