"""
File: /app/ocr/paddleocr.py
Description: Module xử lý OCR sử dụng PaddleOCR (PaddleOCRVL).
"""

import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Any
from PIL import Image

from paddleocr import PaddleOCRVL
from ..helper import PaddleVisualizer
from .ocr import register

logger = logging.getLogger(__name__)

@register("paddleocr")
class PaddleOCR:
    """
    Class xử lý OCR sử dụng PaddleOCR.
    """

    def __init__(self, config=None):
        """
        Khởi tạo PaddleOCR model.
        Args:
            config: Dictionary chứa cấu hình (nếu cần).
        """
        # Khởi tạo pipeline PaddleOCRVL
        # Có thể truyền tham số từ config vào đây nếu cần thiết
        if (config is not None):
            self.pipeline = PaddleOCRVL(**config)
        else:
           self.pipeline = PaddleOCRVL()
           
        self.visualizer = PaddleVisualizer()

    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Xử lý danh sách file và trả về danh sách dictionary gồm:
        - original_file: Đường dẫn file gốc.
        - markdown: Nội dung markdown.
        - annotated_images: Danh sách các ảnh đã annotate (PIL Images).
        """
        results = []
        
        for file_path in file_paths:
            path_obj = Path(file_path)
            if not path_obj.exists():
                logger.error(f"❌ File not found: {path_obj}")
                continue
            
            try:
                # 1. Predict
                # PaddleOCRVL.predict trả về một list các kết quả (thường là 1 cho ảnh, nhiều cho PDF)
                output = self.pipeline.predict(str(path_obj))
                
                full_markdown = ""
                annotated_images = []
                
                # 2. Process results
                for i, res in enumerate(output):
                    if hasattr(res, "markdown"):
                        full_markdown += res.markdown + "\n\n"
                
                # Return original image as annotated image (fallback)
                try:
                    image = Image.open(path_obj).convert("RGB")
                    annotated_images.append({
                        "image": image,
                        "filename": f"{path_obj.stem}_annotated.png"
                    })
                except Exception:
                    pass
                
                # 5. Append result
                results.append({
                    "original_file": path_obj,
                    "markdown": full_markdown.strip(),
                    "annotated_images": annotated_images
                })
                    
            except Exception as e:
                logger.error(f"💥 Error processing {path_obj.name}: {e}")
        
        return results