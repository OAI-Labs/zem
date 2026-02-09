import os
from pathlib import Path
from typing import List, Union, Dict, Any
from dotenv import load_dotenv
from loguru import logger  # <--- Import loguru ở đây

# Imports specific to Agentic Document Extraction
from landingai_ade import LandingAIADE
from landingai_ade.types import ParseResponse, ExtractResponse
# Import helper class
from ..helper import LandingVisualizer
from .ocr import register

# Với loguru, bạn không cần dòng logging.getLogger(__name__) nữa
# logger được import trực tiếp và dùng được ngay

@register("landingade")
class LandingADE:
    """
    Class xử lý OCR sử dụng LandingAI.
    Nhiệm vụ:
    1. Nhận danh sách file.
    2. Gửi request OCR.
    3. Trả về kết quả (parse_result) để pipeline xử lý.
    """

    def __init__(self, model):
        # Load environment variables
        _ = load_dotenv(override=True)
        
        self.client = LandingAIADE()
        self.visualizer = LandingVisualizer()
        self.model = model

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
                # Loguru error
                logger.error(f"❌ File not found: {path_obj}")
                continue
            
            try:
                # 1. Parse Document
                parse_result = self.client.parse(
                    document=path_obj,
                    model=self.model
                )
                
                # 2. Visualize (Generate images in memory, do not save yet)
                # save_files=False -> helper sẽ trả về list ảnh và tên file, không lưu xuống đĩa
                annotated_images = self.visualizer.draw_bounding_boxes_2(
                    groundings=parse_result.grounding,
                    document_path=path_obj,
                    save_files=False
                )
                
                results.append({
                    "original_file": path_obj,
                    "markdown": parse_result.markdown,
                    "annotated_images": annotated_images
                })

            except Exception as e:
                # Loguru error. 
                # Mẹo: Nếu muốn in cả stack trace (truy vết lỗi) đẹp, bạn có thể dùng logger.exception(...)
                logger.error(f"💥 Error processing {path_obj.name}: {e}")
        
        return results