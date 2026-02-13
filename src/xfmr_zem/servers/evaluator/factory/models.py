from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import opik

class HuggingFaceLM:
    def __init__(self, model_id: str):
        self.model_id = model_id
        print(f"Loading HuggingFace Model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Model {model_id} loaded successfully.")

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
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class ModelFactory:
    @staticmethod
    def get_model(engine_type: str, model_id: str) -> Any:
        if engine_type.lower() == "huggingface":
            return HuggingFaceLM(model_id)
        else:
            raise ValueError(f"Unknown model engine: {engine_type}")
