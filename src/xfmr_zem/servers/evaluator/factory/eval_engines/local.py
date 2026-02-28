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
    def __init__(self, model_id: str, model_params: dict, **kwargs):
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
        self.model_params = model_params if model_params is not None else {}
        
        self.model_config = self.model_params.get("model_config", {})
        self.generate_config = self.model_params.get("generate_config", {})
        self._load_model()

    @track(name="load_hf_model")
    def _load_model(self):
        self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.model_config.get("dtype", self._torch.float16),
            device_map=self.model_config.get("device_map", "auto")
        )

        self.generator = self._hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.model_config.get("device_map", "auto")
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
        logger.info(f"\n--- [START GENERATING] ---\n")

        params = self.generate_config.copy()  # Start with default generate_config
        params.update(kwargs)

        try:
            response = self.generator(input, **params)
            raw_text = response[0]["generated_text"].strip()

            if not response_format:
                return raw_text

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
        pass


class OpikLocalFactory:
    @staticmethod
    def create_model(provider: str, model_id: str, model_params: Optional[Dict[str, Any]] = None) -> Any:
        if provider == "huggingface":
            return OpikHFModel(model_id=model_id, model_params=model_params)
        else:
            raise ValueError(f"Unsupported local provider: {provider}")
