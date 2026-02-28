from typing import Any, Optional
import opik
from loguru import logger


class HuggingFaceLM:
    def __init__(self, model_id: str, model_params: dict = None):
        # Lazy import: chỉ import khi thực sự cần (evaluator-local extra)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Thiếu dependencies cho local model. "
                "Hãy cài: pip install 'xfmr-zem[evaluator-local]'"
            )

        self.model_id = model_id
        self.model_params = model_params if model_params is not None else {}
        self.model_config = model_params.get("model_config", {})
        self.generate_config = model_params.get("generate_config", {})

        logger.info(f"Loading HuggingFace Model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.model_config.get("dtype", "auto"),
            device_map=self.model_config.get("device_map", "auto")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            torch_dtype=self.model_config.get("dtype", "auto"),
            device_map=self.model_config.get("device_map", "auto")
        )
        logger.info(f"Model {model_id} loaded successfully.")

    @opik.track
    def generate(self, input_text: str, system_prompt: Optional[str] = None) -> str:
        """
        Causal LM generation: Input text -> Output text.
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=self.generate_config.get("tokenize", False),
            add_generation_prompt=self.generate_config.get("add_generation_prompt", True)
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.generate_config.get("max_new_tokens", 512),
            temperature=self.generate_config.get("temperature", 1.0),
            top_p=self.generate_config.get("top_p", 0.95),
            do_sample=self.generate_config.get("do_sample", True),
            top_k=self.generate_config.get("top_k", 50),
            max_length=self.generate_config.get("max_length", 200),           
            min_length=self.model_params.get("min_length", 5),            
            early_stopping=self.model_params.get("early_stopping", True), 
            num_beams=self.model_params.get("num_beams", 1),
            num_return_sequences=self.model_params.get("num_return_sequences", 1),
            repetition_penalty=self.model_params.get("repetition_penalty", 1.2), 
            no_repeat_ngram_size=self.model_params.get("no_repeat_ngram_size", 3), 
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class vLLM:
    def __init__(self, model_id: str, model_params: dict = None):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "Thiếu dependencies cho local vLLM. "
                "Hãy cài: pip install 'vllm'"
            ) from exc

        self.model_id = model_id
        self.model_params = model_params or {}
        
        self.model_config = self.model_params.get("model_config", {})
        self.generate_config = self.model_params.get("generate_config", {})

        logger.info(f"Loading vLLM model: {model_id}")
        self.llm = LLM(
            model=model_id,
            **self.model_config
        )

    def _build_prompt(self, input_text: str, system_prompt: Optional[str]) -> str:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(input_text)
        return "\n\n".join(parts).strip()

    @opik.track
    def generate(self, input_text: str, system_prompt: Optional[str] = None) -> str:
        prompt = self._build_prompt(input_text, system_prompt or "You are a helpful assistant.")
        sampling_params = SamplingParams(**self.model_params)
        stop_sequences = self.model_params.get("stop_sequences")
        results = self.llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            stop_sequences=stop_sequences
        )

        output_text = ""
        for result in results:
            for output in result.outputs:
                output_text += output.text
            break

        return output_text.strip()

    def __del__(self):
        try:
            self.llm.shutdown()
        except Exception:
            pass
        
class ModelFactory:
    @staticmethod
    def get_model(engine_type: str, model_id: str, model_params: dict = None) -> Any:
        if engine_type.lower() == "huggingface":
            return HuggingFaceLM(model_id, model_params)
        elif engine_type.lower() == "vllm":
            return vLLM(model_id, model_params)
        else:
            raise ValueError(f"Unknown model engine: {engine_type}")
