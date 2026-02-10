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
import xfmr_zem.servers.doc_parser.corrector as corrector
import xfmr_zem.servers.doc_parser.ocr as ocr
from xfmr_zem.servers.doc_parser.helper import MarkdownExtractor
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("ocr ", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def paddle_ocr(
    data: Any,  
) -> Any:
    """Nhận vào danh sách văn bản, tự động chọn GPU rảnh nhất (List Comp) và trả về markdown"""
    
    # # #GPU SETUP
    # logger.info(f"CUDA COUNT DEVICES DEVICES: {torch.cuda.device_count()}")
    # if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    #     try:
    #         gpu_stats = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    #         best_mem, best_idx = max(gpu_stats)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    #         logger.info(f"[PaddleOCR] Selected GPU {best_idx} (Free: {best_mem / 1024**3:.2f} GB)")
    #     except Exception as e:
    #         logger.info(f"[PaddleOCR] Lỗi chọn GPU: {e}")
    # logger.info(f"CUDA COUNT DEVICES AFTER: {torch.cuda.device_count()}")
    # gpu_stats = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    # best_mem, best_idx = max(gpu_stats)
    # logger.info(f"Best info {best_mem} - {best_idx}")
    # # --- BƯỚC 2: Chạy PaddleOCR ---

    items = server.get_data(data)
    pipeline = PaddleOCRVL() # Khởi tạo sau khi đã set CUDA_VISIBLE_DEVICES
    all_results = []
    files = items if isinstance(items, list) else [items]

    for item in files:
        file_path = item
        if isinstance(item, dict):
            for key in item:
                if "path" in key:
                    file_path = item[key]
                    break
        if not os.path.exists(str(file_path)):
            continue
            
        try:
            output = pipeline.predict(str(file_path))
            file_md = ""
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for res in output:
                    res.save_to_markdown(save_path=temp_dir)
                
                md_files = sorted(glob.glob(os.path.join(temp_dir, "*.md")))
                for md_file in md_files:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        file_md += f.read() + "\n\n"
                        
            all_results.append({'markdown': file_md.strip()})
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            all_results.append({'markdown': f"Error processing {file_path}: {e}"})
    return server.save_output(all_results)


@server.tool()
def landingai_ocr(
    data: Any,
    model: str = "dpt-2-latest"
) -> Any:
    """Nhận vào danh sách các văn bản, trả về text markdown dùng LandingAI"""

    items = server.get_data(data)
    model = server.parameters['landingai_ocr']['model']
    _ = load_dotenv(override=True)
    client = LandingAIADE()
    all_results = []
    files = items if isinstance(items, list) else [items]

    for item in files:
        try:
            file_path = item
            if isinstance(item, dict):
                for key in item:
                    if "path" in key:
                        file_path = item[key]
                        break
            parse_result = client.parse(document=Path(file_path), model="dpt-2-latest")
            all_results.append({'markdown': parse_result.markdown})
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            all_results.append({'markdown': f"Error processing {item}: {e}"})
    return server.save_output(all_results)



if (__name__=="__main__"):
    server.run()
