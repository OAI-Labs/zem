import json
import re
from typing import Any, Dict, List, Optional
from opik import track
from pydantic import BaseModel
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

from opik.evaluation.models import OpikBaseModel


class OpikHFModel(OpikBaseModel):
    def __init__(self, model_id: str, **kwargs):
        # Lazy import: chỉ import khi thực sự cần (evaluator-local extra)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
            self._torch = torch
            self._AutoTokenizer = AutoTokenizer
            self._AutoModelForCausalLM = AutoModelForCausalLM
            self._hf_pipeline = hf_pipeline
        except ImportError:
            raise ImportError(
                "Thiếu dependencies cho local model. "
                "Hãy cài: pip install 'xfmr-zem[evaluator-local]'"
            )

        super().__init__(model_name=model_id)
        self.model_id = model_id
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.01)
        self.device = kwargs.get("device", "auto")
        self._load_model()

    @track(name="load_hf_model")
    def _load_model(self):
        self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self._torch.float16,
            device_map=self.device
        )

        self.generator = self._hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device
        )

    def _extract_json_string(self, text: str) -> str:
        """Cắt lấy phần JSON từ output"""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return text[start: end + 1]
        return text

    @logger.catch(reraise=True)
    @track(name="hf_generate_string")
    def generate_string(self, input: str, response_format: Any = None, **kwargs: Any) -> str:
        logger.info(f"\n--- [GENERATE START] ---\n")

        params = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.01 if response_format else self.temperature),
            "do_sample": True,
            "return_full_text": False
        }

        try:
            response = self.generator(input, **params)
            raw_text = response[0]["generated_text"].strip()

            if not response_format:
                return raw_text

            logger.info("--- [VALIDATING JSON] ---")
            json_str = self._extract_json_string(raw_text)
            data = json.loads(json_str)

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                logger.info(f"Validating against Pydantic Model: {response_format.__name__}")
                validated_obj = response_format.model_validate(data)
                final_json = validated_obj.model_dump_json()
                logger.info("VALIDATION SUCCESS ✅")
                return final_json

            logger.info("VALIDATION SUCCESS (Dict) ✅")
            return json.dumps(data, ensure_ascii=False)

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON PARSE ERROR: {e}\nBad String: {json_str}")
            return f'{{"error": "JSONDecodeError", "details": "{str(e)}", "raw_output": "{raw_text}"}}'

        except Exception as e:
            logger.error(f"❌ GENERATION/VALIDATION ERROR: {e}")
            return f'{{"error": "RuntimeError", "details": "{str(e)}"}}'

    @track(name="hf_generate_provider_response")
    def generate_provider_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        prompt = "\n".join([f"{m.get('role','').title()}: {m.get('content','')}" for m in messages])
        prompt += "\nAssistant:"

        generated_text = self.generate_string(
            prompt,
            response_format=kwargs.pop("response_format", None),
            **kwargs
        )

        return {
            "choices": [{"message": {"role": "assistant", "content": generated_text}}],
            "model": self.model_id
        }


class OpikLocalFactory:
    @staticmethod
    def create_model(provider: str, model_id: str, **kwargs) -> Any:
        if provider == "huggingface":
            return OpikHFModel(model_id=model_id, **kwargs)
        else:
            raise ValueError(f"Unsupported local provider: {provider}")