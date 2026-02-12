from typing import List, Dict, Any, Callable, Optional, Union
import json
from loguru import logger
from .schemas import MultipleChoiceItem, TextGenerationItem

class BaseDataset:
    def __init__(
        self, 
        dataset_path: str, 
        limit: int = 100,
        field_mapping: Optional[Dict[str, str]] = None
    ):
        self.dataset_path = dataset_path
        self.limit = limit
        self.field_mapping = field_mapping

    def load_data(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:   
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            raise e

        if not isinstance(data, list):
             raise ValueError("Dataset file must contain a JSON list of items.")

        # Limit
        if self.limit and len(data) > self.limit:
            data = data[:self.limit]

        # Transform and Validate
        transformed_data = []
        for item in data:
            if self.field_mapping:
                for target, source in self.field_mapping.items():
                    if source in item:
                        item[target] = item[source]

            if self.validate(item):
                transformed_data.append(item)
            else:
                logger.warning(f"Item failed validation: {item}")
        
        return transformed_data

    def validate(self, item: Dict[str, Any]) -> bool:
        """
        Base validation. Override in subclasses.
        """
        return True

class MultipleChoiceDataset(BaseDataset):
    def __init__(
        self, 
        dataset_path: str, 
        limit: int = 100,
        input_field: str = "input",
        choices_field: str = "choices",
        reference_field: str = "output"
    ):
        mapping = {
            "input": input_field,
            "choices": choices_field,
            "output": reference_field
        }
        super().__init__(dataset_path, limit, field_mapping=mapping)
        self.type = "multiple_choice"

    def validate(self, item: Dict[str, Any]) -> bool:
        # Check against MultipleChoiceItem schema: input, choices, output
        required = ["input", "choices", "output"]
        missing = [k for k in required if k not in item]
        if missing:
             logger.debug(f"Missing keys for MultipleChoiceDataset: {missing}")
             return False
        return True

class TextGenerationDataset(BaseDataset):
    def __init__(
        self, 
        dataset_path: str, 
        limit: int = 100,
        input_field: str = "input",
        context_field: str = "context",
        reference_field: str = "output"
    ):
        mapping = {
            "input": input_field,
            "context": context_field,
            "output": reference_field
        }
        super().__init__(dataset_path, limit, field_mapping=mapping)
        self.type = "text_generation"

    def validate(self, item: Dict[str, Any]) -> bool:
        # Check against TextGenerationItem schema: input, output. context is optional
        required = ["input", "context", "output"]
        missing = [k for k in required if k not in item]
        if missing:
             logger.debug(f"Missing keys for TextGenerationDataset: {missing}")
             return False
        return True

class DatasetFactory:
    @staticmethod
    def get_dataset(
        dataset_type: str, 
        dataset_path: str, 
        limit: int = 100,
        input_field: str = "input",
        choices_field: str = "choices",
        reference_field: str = "output",
        context_field: str = "context",
        **kwargs # Ignore extra kwargs passed if any
    ) -> BaseDataset:
        if dataset_type == "multiple_choice":
            return MultipleChoiceDataset(
                dataset_path, 
                limit,
                input_field=input_field,
                choices_field=choices_field,
                reference_field=reference_field
            )
        elif dataset_type == "text_generation":
            return TextGenerationDataset(
                dataset_path, 
                limit,
                input_field=input_field,
                context_field=context_field,
                reference_field=reference_field
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
