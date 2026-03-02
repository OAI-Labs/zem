import json
import re
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from opik import track
from pydantic import BaseModel
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

from opik.evaluation.models import OpikBaseModel


class OpikLocalBaseModel(OpikBaseModel, ABC):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

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

    @abstractmethod
    @track(name="generate for model evaluation")
    def generate(self, input: str, **kwargs) -> str:
        """Hàm sinh text nội bộ cho từng model engine"""
        pass

    @logger.catch(reraise=True)
    @track(name="generate_string")
    def generate_string(self, input: str, response_format: Any = None, **kwargs: Any) -> str:
        try:
            # Gọi hàm sinh text của lớp con
            raw_text = self.generate(input, **kwargs)

            if not response_format:
                return raw_text

            json_str = self._extract_json_string(raw_text)
            
            # Parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON PARSE ERROR: {e}\nBad String: {json_str}")
                return f'{{"error": "JSONDecodeError", "details": "{str(e)}", "raw_output": "{raw_text}"}}'

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                logger.info(f"Validating against Pydantic Model: {response_format.__name__}")
                validated_obj = response_format.model_validate(data)
                final_json = validated_obj.model_dump_json()
                logger.info("VALIDATION SUCCESS ✅")
                return final_json

            logger.info("VALIDATION SUCCESS (Dict) ✅")
            return json.dumps(data, ensure_ascii=False)

        except Exception as e:
            logger.error(f"❌ GENERATION/VALIDATION ERROR: {e}")
            return f'{{"error": "RuntimeError", "details": "{str(e)}"}}'

    @track(name="generate_provider_response")
    def generate_provider_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        pass


class OpikHFModel(OpikLocalBaseModel):
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
        self.tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_id,
            torch_dtype=self.model_config.get("dtype", "auto"),
            device_map=self.model_config.get("device_map", "auto"),
            cache_dir=self.model_config.get("cache_dir", None)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.model_config.get("dtype", self._torch.float16),
            device_map=self.model_config.get("device_map", "auto"),
            cache_dir=self.model_config.get("cache_dir", None)
        )

        self.generator = self._hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.model_config.get("device_map", "auto")
        )

    def generate(self, input: str, **kwargs) -> str:
        params = self.generate_config.copy()
        params.update(kwargs)
        
        response = self.generator(input, **params)
        return response[0]["generated_text"].strip()


class OpikVLLMModel(OpikLocalBaseModel):
    def __init__(self, model_id: str, model_params: dict, **kwargs):
        try:
            from vllm import LLM, SamplingParams
            self._LLM = LLM
            self._SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "Thiếu dependencies cho vLLM model. "
                "Hãy cài: pip install 'xfmr-zem[evaluator-vllm]'"
            )

        super().__init__(model_name=model_id)
        self.model_id = model_id
        self.model_params = model_params or {}
        
        self.model_config = self.model_params.get("model_config", {})
        self.generate_config = self.model_params.get("generate_config", {})
        
        self._load_model()

    @track(name="load_vllm_model")
    def _load_model(self):
        logger.info(f"Loading vLLM model: {self.model_id}")
        self.llm = self._LLM(
            model=self.model_id,
            **self.model_config
        )
        logger.info(f"Finished Loading vLLM model: {self.model_id}")

    def generate(self, input: str, **kwargs) -> str:
        params = self.generate_config.copy()
        params.update(kwargs)
        
        messages = [
            {"role": "user", "content": input}
        ]
        
        sampling_params = self._SamplingParams(**params)
        
        results = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        output_text = ""
        for result in results:
            for output in result.outputs:
                output_text += output.text
            break
            
        return output_text.strip()


class OpikLocalFactory:
    @staticmethod
    def create_model(provider: str, model_id: str, model_params: Optional[Dict[str, Any]] = None) -> Any:
        if provider == "huggingface":
            return OpikHFModel(model_id=model_id, model_params=model_params)
        elif provider == "vllm":
            return OpikVLLMModel(model_id=model_id, model_params=model_params)
        else:
            raise ValueError(f"Unsupported local provider: {provider}")
