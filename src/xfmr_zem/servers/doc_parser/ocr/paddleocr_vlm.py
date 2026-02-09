"""
File: /app/ocr/paddleocr.py
Description: Module xử lý OCR sử dụng PaddleOCR (PaddleOCRVL).
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Union, Dict, Any
from PIL import Image

from paddleocr import PaddleOCRVL
from helper import PaddleVisualizer

try:
    from .ocr import register
except:
    from ocr import register

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
        
        # Tạo thư mục output cho JSON nếu chưa tồn tại
        
        for file_path in file_paths:
            path_obj = Path(file_path)
            if not path_obj.exists():
                logger.error(f"❌ File not found: {path_obj}")
                continue
            
            try:
                # Sử dụng TemporaryDirectory để lưu kết quả trung gian từ PaddleOCR
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 1. Predict
                    # PaddleOCRVL.predict trả về một list các kết quả (thường là 1 cho ảnh, nhiều cho PDF)
                    output = self.pipeline.predict(str(path_obj))
                    
                    full_markdown = ""
                    annotated_images = []
                    
                    # 2. Process results
                    for i, res in enumerate(output):

                        json_path = Path(temp_dir) / "temp.json"
                        res.save_to_json(save_path=json_path)

                        # Lưu markdown và ảnh vào thư mục tạm
                        # PaddleOCR sẽ tự động đặt tên file dựa trên input hoặc mặc định
                        res.save_to_markdown(save_path=temp_dir)
                        
                        # Sử dụng PaddleVisualizer để vẽ bounding box
                        # Dự đoán tên file JSON (thường là {stem}_res.json)
                        
                        if json_path.exists():
                            viz_img = self.visualizer.draw_bounding_boxes(str(path_obj), str(json_path))
                            fname = f"{path_obj.stem}_annotated.png"
                            if len(output) > 1:
                                fname = f"{path_obj.stem}_page_{i+1}_annotated.png"
                            annotated_images.append({"image": viz_img, "filename": fname})
                    
                    # 3. Read back Markdown
                    # Tìm tất cả file .md trong temp_dir
                    md_files = sorted(list(Path(temp_dir).glob("*.md")))
                    for md_file in md_files:
                        text = md_file.read_text(encoding='utf-8')
                        full_markdown += text + "\n\n"
                    
                    # 5. Append result
                    results.append({
                        "original_file": path_obj,
                        "markdown": full_markdown.strip(),
                        "annotated_images": annotated_images
                    })
                    
            except Exception as e:
                logger.error(f"💥 Error processing {path_obj.name}: {e}")
        
        return results

if (__name__ == "__main__"):
    # Test nhanh PaddleOCR class
    ocr = PaddleOCR()
    test_image = "data/save/law/Screenshot 2026-02-06 182808.png"  # Thay bằng đường dẫn ảnh thật để test
    results = ocr.process_batch([test_image])
    for res in results:
        print("Markdown Output:")
        for key, value in res.items():
            print(key, value)
    print(f"Number of Annotated Images: {len(res['annotated_images'])}")