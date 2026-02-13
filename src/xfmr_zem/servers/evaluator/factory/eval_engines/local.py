import torch
import json
import re
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from opik import track
from pydantic import BaseModel
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="INFO")
# Giả lập OpikBaseModel

from opik.evaluation.models import OpikBaseModel

class OpikHFModel(OpikBaseModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_name=model_id)
        self.model_id = model_id
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        # Temperature thấp để model tập trung vào logic chấm điểm, không sáng tạo lung tung
        self.temperature = kwargs.get("temperature", 0.01) 
        self.device = kwargs.get("device", "auto")
        self._load_model()
    
    @track(name="load_hf_model")
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device
        )

    def _extract_json_string(self, text: str) -> str:
        """Cắt lấy phần JSON từ output"""
        text = text.strip()
        # Regex tìm ```json { ... } ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        
        # Fallback: tìm { }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1: return text[start : end + 1]
        return text

    @logger.catch(reraise=True)
    @track(name="hf_generate_string")
    def generate_string(self, input: str, response_format: Any = None, **kwargs: Any) -> str:
        # 1. LOG INPUT
        # Quan sát xem input thực tế nhận vào là gì (đã có schema hay chưa?)
        logger.info(f"\n--- [GENERATE START] ---\n")

        # 2. GENERATION
        # Không can thiệp sửa input nữa, chỉ setup tham số chạy
        params = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", 0.01 if response_format else self.temperature), # Temp thấp nếu cần JSON
            "do_sample": True,
            "return_full_text": False
        }

        try:
            # Gọi model sinh text
            response = self.generator(input, **params)
            raw_text = response[0]["generated_text"].strip()
            
            # Nếu không yêu cầu format đặc biệt, trả về luôn
            if not response_format:
                return raw_text

            # 3. CHECK FORMAT (VALIDATION)
            logger.info("--- [VALIDATING JSON] ---")
            
            # Bước A: Extract JSON từ text (lọc rác markdown)
            json_str = self._extract_json_string(raw_text)
            
            # Bước B: Parse & Validate
            data = json.loads(json_str) # Thử parse JSON thuần
            
            # Nếu có Pydantic Model, validate chặt chẽ kiểu dữ liệu
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                logger.info(f"Validating against Pydantic Model: {response_format.__name__}")
                validated_obj = response_format.model_validate(data)
                
                # Thành công!
                final_json = validated_obj.model_dump_json()
                logger.info("VALIDATION SUCCESS ✅")
                return final_json
            
            # Nếu chỉ là dict schema thường
            logger.info("VALIDATION SUCCESS (Dict) ✅")
            return json.dumps(data, ensure_ascii=False)

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON PARSE ERROR: {e}\nBad String: {json_str}")
            # Trả về lỗi dạng JSON để Opik ghi nhận thay vì crash chương trình
            return f'{{"error": "JSONDecodeError", "details": "{str(e)}", "raw_output": "{raw_text}"}}'
            
        except Exception as e:
            logger.error(f"❌ GENERATION/VALIDATION ERROR: {e}")
            return f'{{"error": "RuntimeError", "details": "{str(e)}"}}'

    @track(name="hf_generate_provider_response")
    def generate_provider_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        # Chuyển messages thành prompt string cơ bản
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