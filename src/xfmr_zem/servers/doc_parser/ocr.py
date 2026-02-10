"""
File: /app/xfmr-zem/src/xfmr_zem/servers/doc_parser/server.py
Description: OCR processing pipeline as a function.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import glob
import tempfile
import torch
import torch.nn as nn
from paddleocr import PaddleOCRVL
from landingai_ade import LandingAIADE
from dotenv import load_dotenv
#Import Zem
from xfmr_zem.server import ZemServer

# Import OCR Processor
from loguru import logger
from tqdm import tqdm

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("ocr", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def paddle_ocr(
    data: Any, 
    field: str = "file_path"  # Trường chứa đường dẫn file trong input items
) -> Any:
    """
    Nhận vào danh sách items, lấy đường dẫn file từ `field`, chạy OCR và trả về markdown.
    """
    
    # # #GPU SETUP (Giữ nguyên logic comment của bạn)
    # logger.info(f"CUDA COUNT DEVICES DEVICES: {torch.cuda.device_count()}")
    # if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    #     try:
    #         gpu_stats = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    #         best_mem, best_idx = max(gpu_stats)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    #         logger.info(f"[PaddleOCR] Selected GPU {best_idx} (Free: {best_mem / 1024**3:.2f} GB)")
    #     except Exception as e:
    #         logger.info(f"[PaddleOCR] Lỗi chọn GPU: {e}")

    items = server.get_data(data)
    items = items if isinstance(items, list) else [items]
    
    # Khởi tạo pipeline một lần bên ngoài vòng lặp
    try:
        pipeline = PaddleOCRVL() 
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        # Nếu init lỗi thì trả về lỗi cho tất cả items
        return server.save_output([{"markdown": f"Error init model: {e}"} for _ in items])

    all_results = []

    for item in tqdm(items, desc="PaddleOCR Processing"):
        try:
            # 1. Lấy đường dẫn file từ field
            if isinstance(item, dict) and field in item:
                file_path = item[field]
            else:
                # Nếu input là string thì coi đó là path luôn (tùy chọn), 
                # hoặc raise lỗi nếu muốn strict theo yêu cầu "chỉ xử lý field"
                # Ở đây tôi raise lỗi để đúng logic try-except bạn yêu cầu
                raise KeyError(f"Field '{field}' not found or item is not a dict")

            # 2. Kiểm tra file tồn tại
            if not os.path.exists(str(file_path)):
                logger.warning(f"File not found: {file_path}")
                all_results.append({'markdown': ""})
                continue
            
            # 3. Chạy Inference
            output = pipeline.predict(str(file_path))
            file_md = ""
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save markdown ra temp rồi đọc lại
                for res in output:
                    res.save_to_markdown(save_path=temp_dir)
                
                md_files = sorted(glob.glob(os.path.join(temp_dir, "*.md")))
                for md_file in md_files:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        file_md += f.read() + "\n\n"
                        
            all_results.append({'markdown': file_md.strip()})

        except Exception as e:
            logger.warning(f"Error processing item with PaddleOCR: {e}")
            # Trả về chuỗi rỗng hoặc thông báo lỗi tùy nhu cầu
            all_results.append({'markdown': ""})

    return server.save_output(all_results)


@server.tool()
def landingai_ocr(
    data: Any,
    field: str = "file_path", # Trường chứa đường dẫn file
    model: str = "dpt-2-latest" # Tham số model (có thể bị override bởi parameters.yml)
) -> Any:
    """
    Nhận vào danh sách items, lấy đường dẫn file từ `field`, trả về text markdown dùng LandingAI.
    """

    items = server.get_data(data)
    items = items if isinstance(items, list) else [items]
    
    # Load config model từ parameters (ưu tiên config file)
    try:
        param_model = server.parameters['landingai_ocr']['model']
        if param_model:
            model = param_model
    except KeyError:
        pass # Dùng default argument

    _ = load_dotenv(override=True)
    client = LandingAIADE()
    all_results = []

    for item in tqdm(items, desc="LandingAI Processing"):
        try:
            # 1. Lấy đường dẫn file từ field
            if isinstance(item, dict) and field in item:
                file_path = item[field]
            else:
                raise KeyError(f"Field '{field}' not found in item")

            # 2. Gọi API LandingAI
            # Lưu ý: Client LandingAI có thể tự handle việc check file path
            parse_result = client.parse(document=Path(file_path), model=model)
            
            all_results.append({'markdown': parse_result.markdown})

        except Exception as e:
            logger.warning(f"Error processing item with LandingAI: {e}")
            all_results.append({'markdown': ""})

    return server.save_output(all_results)


if (__name__=="__main__"):
    server.run()