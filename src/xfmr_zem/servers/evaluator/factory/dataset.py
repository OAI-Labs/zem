from typing import List, Dict, Any, Optional, Union
import json
from loguru import logger

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
            raise

        if not isinstance(data, list):
             raise ValueError("Dataset file must contain a JSON list of items.")

        # Limit
        if self.limit and len(data) > self.limit:
            data = data[:self.limit]

        transformed_data = []
        for item in data:
            # 1. Mapping: Đưa các trường từ file về tên chuẩn (input, context, reference)
            if self.field_mapping:
                for standard_key, file_key in self.field_mapping.items():
                    if file_key in item:
                        item[standard_key] = item[file_key]

            # 2. Process: Xử lý logic tạo prompt
            processed_item = self.process_item(item)
            
            # 3. Validate: Kiểm tra đầu ra cuối cùng
            if processed_item and self.validate(processed_item):
                transformed_data.append(processed_item)
            else:
                logger.warning(f"Item failed validation or processing: {item}")
        
        logger.info(f"Successfully loaded and transformed {len(transformed_data)} items.")
        return transformed_data

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
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
        limit: int = 100,
        input_field: str = "input", 
        choices_field: str = "choices", 
        reference_field: str = "output",
        context_field: str = "context",
    ):
        # Mapping thẳng về tên chuẩn như bạn yêu cầu
        mapping = {
            "input": input_field,
            "choices": choices_field,
            "reference": reference_field,
            "context": context_field
        }
        self.task_instruction = "The model is in a multiple-choice task. Please select the correct answer and output it in the format ['index']. Index starts from 0."
        super().__init__(dataset_path, limit, field_mapping=mapping)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Lấy dữ liệu bằng đúng key chuẩn
        question = item.get("input", "")
        choices = item.get("choices", [])
        context = item.get("context", "")
        reference = item.get("reference")

        # Xử lý Context
        context_list = []
        if context:
            if isinstance(context, list):
                context_list.extend([str(c) for c in context])
            else:
                context_list.append(str(context))
        
        context_list.append(self.task_instruction)

        # Xử lý Input (Tạo prompt)
        if not choices:
            logger.warning("Multiple choice item missing choices.")
            return None

        choices_str = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
        final_prompt = (
            f"{self.task_instruction}\n\n"
            f"Context: {context if context else 'None'}\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}\n"
            f"Answer:"
        )
        
        # Xử lý Output
        final_output = f"['{reference}']" if reference is not None else "['']"

        return {
            "input": final_prompt,
            "context": context_list,
            "reference": final_output
        }


class TextGenerationDataset(BaseDataset):
    def __init__(
        self, 
        dataset_path: str, 
        limit: int = 100,
        input_field: str = "input", 
        reference_field: str = "output", 
        context_field: str = "context",
    ):
        # Mapping thẳng về tên chuẩn
        mapping = {
            "input": input_field,
            "context": context_field,
            "reference": reference_field,
        }
        self.task_instruction = "The model is in a text generation task and must generate a response based on the context."
        super().__init__(dataset_path, limit, field_mapping=mapping)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Lấy dữ liệu bằng đúng key chuẩn
        question = item.get("input", "")
        context = item.get("context", "")
        reference = item.get("reference", "")

        # Xử lý Context
        context_list = []
        if context:
            if isinstance(context, list):
                context_list.extend([str(c) for c in context])
            else:
                context_list.append(str(context))
        
        context_list.append(self.task_instruction)

        # Xử lý Input (Tạo prompt)
        context_str = "\n".join(context_list[:-1]) 
        final_prompt = f"System: {self.task_instruction}\nContext: {context_str}\nQuestion: {question}\nAnswer:"

        # Xử lý Output
        final_output = str(reference)

        return {
            "input": final_prompt,
            "context": context_list,
            "reference": final_output
        }


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
        **kwargs
    ) -> BaseDataset:
        if dataset_type == "multiple_choice":
            return MultipleChoiceDataset(
                dataset_path=dataset_path, 
                limit=limit,
                input_field=input_field,
                choices_field=choices_field,
                reference_field=reference_field,
                context_field=context_field
            )
        elif dataset_type == "text_generation":
            return TextGenerationDataset(
                dataset_path=dataset_path, 
                limit=limit,
                input_field=input_field,
                reference_field=reference_field,
                context_field=context_field
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")