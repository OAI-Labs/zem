import os
import sys
import base64
import tempfile
import uuid
from typing import Any, List, Dict, Tuple
# [ADD] Import thư viện xử lý đa luồng
from concurrent.futures import ThreadPoolExecutor, as_completed

from xfmr_zem.server import ZemServer
from xfmr_zem.servers.ocr.engines import OCREngineFactory
from loguru import logger

# Initialize ZemServer for OCR
server = ZemServer("ocr")
logger.remove() # Remove default handler to avoid duplicate logs if configured elsewhere
logger.add(sys.stderr, level="DEBUG")

def process_item(
    input_data: str,
    mode: str,
    ocr_engine,
    engine: str,
    model_id: str | None,
    field: str,
    unique_temp_dir: str,
) -> List[Dict[str, Any]]:
    # ... (Giữ nguyên logic hàm process_item như cũ) ...
    if not input_data:
        return []

    if ocr_engine is None:
        logger.error("Failed to initialize OCR engine.")
        return []

    file_path_to_process = None
    temp_file_created = False

    try:
        if mode == "file_path":
            if not os.path.exists(input_data):
                logger.warning(f"File not found: {input_data}")
                return []
            if str(input_data).lower().endswith(".pdf"):
                logger.warning(f"PDF is not supported anymore. Skipping: {input_data}")
                return []
            file_path_to_process = input_data
        
        elif mode == "base64":
            try:
                img_bytes = base64.b64decode(input_data)
                # UUID đảm bảo tên file không bị trùng khi chạy đa luồng
                temp_img_name = f"image_{uuid.uuid4()}.png" 
                file_path_to_process = os.path.join(unique_temp_dir, temp_img_name)
                with open(file_path_to_process, "wb") as f:
                    f.write(img_bytes)
                temp_file_created = True
            except Exception as e:
                logger.error(f"Failed to decode/save base64 image: {e}")
                return []
        else:
            logger.warning(f"Unsupported mode: {mode}")
            return []

        try:
            result = ocr_engine.process(file_path_to_process)
            text = result.get("text", "") if isinstance(result, dict) else ""
            engine_name = result.get("engine", engine) if isinstance(result, dict) else engine
        except Exception as e:
            logger.error(f"Failed to process image {file_path_to_process}: {e}")
            text = ""
            engine_name = f"{engine}_error"

        metadata = {"model_id": model_id, "field": field}
        if mode == "file_path":
            metadata["file_path"] = input_data
        else:
            metadata["source"] = "base64"

        return [{"text": text, "engine": engine_name, "metadata": metadata}]
    finally:
        if temp_file_created and file_path_to_process and os.path.exists(file_path_to_process):
            os.remove(file_path_to_process)

@server.tool()
def extract_text(
    data: Any,
    mode: str = "file_path",
    field: str = "file_path",
    engine: str = "paddle_vl", 
    model_id: str = None,
    temp_dir: str = "/tmp",
    max_workers: int = 1, # [ADD] Tham số điều chỉnh số luồng
) -> Any:
    """
    Extracts text from an image using the specified OCR engine with Multi-threading.
    Args:
        mode: Mode to resolve (path or base64)
        field: chosen field to perform task on.
        engine: types used.
        model_id: model used.
    """
    items = server.get_data(data)
    items = items if isinstance(items, list) else [items]

    if not items:
        return server.save_output([])

    try:
        all_results: List[Dict[str, Any]] = []
        os.makedirs(temp_dir, exist_ok=True)
        
        # Khởi tạo Engine một lần (lưu ý: Engine phải Thread-safe hoặc stateless)
        ocr_engine = OCREngineFactory.get_engine(engine, model_id=model_id)

        # Tạo thư mục tạm thời
        with tempfile.TemporaryDirectory(dir=temp_dir) as unique_temp_dir:
            
            # [ADD] Bắt đầu block xử lý đa luồng
            logger.info(f"Starting OCR extraction with {max_workers} threads...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # 1. Submit tasks
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    values = item.get(field)
                    if not values:
                        continue

                    if not isinstance(values, (list, tuple)):
                        values = [values]

                    for value in values:
                        if not value:
                            continue
                        
                        # Đẩy task vào pool
                        future = executor.submit(
                            process_item,
                            input_data=value,
                            mode=mode,
                            ocr_engine=ocr_engine,
                            engine=engine,
                            model_id=model_id,
                            field=field,
                            unique_temp_dir=unique_temp_dir
                        )
                        futures.append(future)

                # 2. Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            all_results.extend(result)
                    except Exception as exc:
                        logger.error(f"Thread generated an exception: {exc}")

        return server.save_output(all_results)
        
    except Exception as e:
        logger.error(f"OCR Error with {engine}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"OCR failed: {str(e)}")

if __name__ == "__main__":
    server.run()