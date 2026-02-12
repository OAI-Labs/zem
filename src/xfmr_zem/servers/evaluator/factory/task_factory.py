from typing import Dict, Any
import re
from .schemas import MCQ_SYSTEM_PROMPT, TEXTGEN_SYSTEM_PROMPT

class BaseTask:
    def __init__(self, model: Any):
        self.model = model

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MultipleChoiceTask(BaseTask):
    def _parse_mcq_output(self, response: str) -> str:
        match = re.search(r"\['(\d+)'\]", response)
        if match:
            return match.group(0)
        return "['']"

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Safely access item fields
        question = item.get("input", "")
        choices = item.get("choices", [])

        # Format input for the model
        # Ensure choices are formatted nicely in the prompt with indices
        choices_str = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])

        prompt = f"Question: {question}\nChoices:\n{choices_str}\nPlease select the correct answer and output it in the format ['index']. Index starts from 0. \nAnswer:"
        system_prompt = item.get("system_prompt") or MCQ_SYSTEM_PROMPT

        generated_response = self.model.generate(prompt, system_prompt=system_prompt)
        parsed_output = self._parse_mcq_output(generated_response)

        reference = item.get("output")
        formatted_reference = f"['{reference}']" if reference is not None else "['']"

        return {
            "output": parsed_output,
            "reference": formatted_reference,
            "expected_output": formatted_reference,
        }

class TextGenerationTask(BaseTask):
    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Safely access item fields
        context = item.get("context", "")
        question = item.get("input", "")

        # Format input for the model
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        system_prompt = item.get("system_prompt") or TEXTGEN_SYSTEM_PROMPT
        generated_response = self.model.generate(prompt, system_prompt=system_prompt)
        
        return {
            "input": question,
            "output": generated_response,
            "context": [context], # Opik metrics often expect context as a list
            "reference": item.get("expected_output"),
            "expected_output": item.get("expected_output")
        }

class TaskFactory:
    @staticmethod
    def get_task(dataset_type: str, model: Any) -> BaseTask:
        if dataset_type == "multiple_choice":
            return MultipleChoiceTask(model)
        elif dataset_type == "text_generation":
            return TextGenerationTask(model)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
