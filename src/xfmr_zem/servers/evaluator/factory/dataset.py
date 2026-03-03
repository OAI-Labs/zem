from typing import List, Dict, Any, Optional
import json
from loguru import logger

class BaseDataset:
    def __init__(self, dataset_path: str, limit: Optional[int] = None):
        self.dataset_path = dataset_path
        self.limit = limit

    def load_data(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:   
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            raise

        if not isinstance(data, list):
             raise ValueError("Dataset file must contain a JSON list of items.")

        if isinstance(self.limit, int) and self.limit > 0 and len(data) > self.limit:
            data = data[:self.limit]

        transformed_data = []
        for index, item in enumerate(data):
            processed_item = self.process_item(item, index)
            
            # 3. Validate: Kiểm tra đầu ra cuối cùng
            if processed_item and self.validate(processed_item):
                transformed_data.append(processed_item)
            else:
                logger.warning(f"Item failed validation or processing: {item}")
        
        logger.info(f"Successfully loaded and transformed {len(transformed_data)} items.")
        return transformed_data

    def process_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Hàm này bắt buộc phải trả về dict có 3 key:
        - input
        - context
        - reference
        """
        raise NotImplementedError

    def validate(self, item: Dict[str, Any]) -> bool:
        required = ["input", "context", "reference"]
        missing = [k for k in required if k not in item]
        
        if missing:
            logger.debug(f"Missing keys in processed item: {missing}")
            return False
            
        if not isinstance(item["context"], list):
            logger.debug(f"Context must be a list, got {type(item['context'])}")
            return False
            
        return True

class MultipleChoiceDataset(BaseDataset):
    def __init__(
        self, 
        dataset_path: str, 
        limit: Optional[int] = None,
        task_instruction: Optional[str] = None,
        index: Optional[int] = 1,
    ):
        default_instruction = "The model is in a multiple-choice task. Please select the correct answer and output it in the format ['index']. Index starts from 0."
        self.task_instruction = task_instruction or default_instruction
        self.choice_index = index
        super().__init__(dataset_path, limit)

    def process_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        try:
            question = item["input"]
            choices = item["choices"]
            reference = item["output"]
        except KeyError as exc:
            raise ValueError(
                f"Dataset item at index {index} missing required key '{exc.args[0]}'. Dữ liệu chưa đúng chuẩn."
            ) from exc

        context = item.get("context", "")

        context_list = []
        if context:
            if isinstance(context, list):
                context_list.extend([str(c) for c in context])
            else:
                context_list.append(str(context))

        context_list.append(self.task_instruction)

        choice_lines = []

        if isinstance(choices, dict):
            for key, value in choices.items():
                choice_lines.append(f"{key}. {value}")
        elif isinstance(choices, list):
            if self.choice_index is None:
                for option in choices:
                    choice_lines.append(str(option))
            else:
                start_point = 0 if self.choice_index == 0 else 1
                for offset, option in enumerate(choices):
                    choice_lines.append(f"{start_point + offset}. {option}")
        else:
            raise ValueError(
                f"Invalid type for choices at index {index}: expected list or dict, got {type(choices)}"
            )

        if not choice_lines:
            raise ValueError(f"Dataset item at index {index} contains no choices.")

        choices_str = "\n".join(choice_lines)

        final_prompt = (
            f"{self.task_instruction}\n\n"
            f"Context: {context if context else 'None'}\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}\n"
            f"Answer:"
        )

        final_output = f"['{reference}']"

        return {
            "input": final_prompt,
            "context": context_list,
            "reference": final_output,
            "order": index
        }


class TextGenerationDataset(BaseDataset):
    def __init__(
        self, 
        dataset_path: str, 
        limit: Optional[int] = None,
        task_instruction: Optional[str] = None,
    ):
        # Mapping thẳng về tên chuẩn
        default_instruction = "The model is in a text generation task and must generate a response based on the context."
        self.task_instruction = task_instruction or default_instruction
        super().__init__(dataset_path, limit)

    def process_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        try:
            question = item["input"]
            reference = item["output"]
        except KeyError as exc:
            raise ValueError(
                f"Dataset item at index {index} missing required key '{exc.args[0]}'. Dữ liệu chưa đúng chuẩn."
            ) from exc
        context = item.get("context", "")

        context_list = []
        if context:
            if isinstance(context, list):
                context_list.extend([str(c) for c in context])
            else:
                context_list.append(str(context))

        context_list.append(self.task_instruction)

        context_str = "\n".join(context_list[:-1])
        final_prompt = f"System: {self.task_instruction}\nContext: {context_str}\nQuestion: {question}\nAnswer:"

        return {
            "input": final_prompt,
            "context": context_list,
            "reference": str(reference),
            "order": index
        }


class DatasetFactory:
    @staticmethod
    def get_dataset(
        dataset_type: str, 
        dataset_path: str, 
        limit: Optional[int] = None,
        task_instruction: Optional[str] = None,
        index: Optional[int] = 1,
        **kwargs
    ) -> BaseDataset:
        if dataset_type == "multiple_choice":
            return MultipleChoiceDataset(
                dataset_path=dataset_path, 
                limit=limit,
                task_instruction=task_instruction,
                index=index
            )
        elif dataset_type == "text_generation":
            return TextGenerationDataset(
                dataset_path=dataset_path, 
                limit=limit,
                task_instruction=task_instruction
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
