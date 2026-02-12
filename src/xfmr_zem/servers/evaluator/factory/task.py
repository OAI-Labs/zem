from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GenerativeTask:
    """
    GenerativeTask is responsible for executing the model on the input to produce the result.
    It isolates the execution logic and standardizes the output format.
    """
    def __init__(self, model: Any):
        self.model = model

    def run(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model generation and returns the structured result.
        
        Args:
            item: Dictionary containing 'input', 'task_context', 'context', 'reference'.
            
        Returns:
            Dictionary with keys: input, output, context, reference, reference.
        """
        input_text = item.get("input", "")
        context = item.get("context", None)
        reference = item.get("reference", None)
        try:
            # Execute model generation
            output = self.model.generate(input_text)
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            output = ""

        return {
            "input": input_text,
            "output": output,
            "context": context,
            "expected_output": reference,
            "reference": reference,  # Often used as an alias for reference
        }

class TaskFactory:
    @staticmethod
    def get_task(task_type: str, model: Any) -> Any:
        if task_type in ["generative"]:
            return GenerativeTask(model)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
