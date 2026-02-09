import torch
import logging
from pathlib import Path
from typing import List, Union, Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .ocr import register

logger = logging.getLogger(__name__)

@register("glmocr")
class GLMOCR:
    """
    Class xử lý OCR sử dụng GLM-OCR.
    """

    def __init__(self, model_path=None):
        """
        Khởi tạo GLM-OCR model.
        Args:
            config: Dictionary chứa cấu hình (nếu cần).
        """
        if model_path is None:
            model_path = "zai-org/GLM-OCR"
            
        self.model_path = model_path
        
        logger.info(f"🚀 Loading GLM-OCR model from: {self.model_path}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Xử lý danh sách file và trả về danh sách dictionary gồm:
        - original_file: Đường dẫn file gốc.
        - markdown: Nội dung markdown.
        - annotated_images: Danh sách ảnh (dùng ảnh gốc nếu không visualize được).
        """
        results = []
        
        for file_path in file_paths:
            path_obj = Path(file_path)
            if not path_obj.exists():
                logger.error(f"❌ File not found: {path_obj}")
                continue
            
            try:
                image = Image.open(path_obj).convert("RGB")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": str(path_obj)
                            },
                            {
                                "type": "text",
                                "text": "Text Recognition:",
                                "formula": "Formula Recognition:",
                                "table": "Table Recognition:"                                
                            }
                        ],
                    }
                ]
                
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                inputs.pop("token_type_ids", None)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
                    
                output_text = self.processor.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=False
                )
                
                # Vì model không hỗ trợ visualize bounding box, ta trả về ảnh gốc
                annotated_images = [{
                    "image": image,
                    "filename": f"{path_obj.stem}_annotated.png"
                }]
                
                results.append({
                    "original_file": path_obj,
                    "markdown": output_text,
                    "annotated_images": annotated_images
                })
                
            except Exception as e:
                logger.error(f"💥 Error processing {path_obj.name}: {e}")
        
        return results
