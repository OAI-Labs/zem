import os
import io
from typing import Any, List

import pandas as pd
from xfmr_zem.server import ZemServer
from xfmr_zem.servers.ocr.engines import OCREngineFactory
from loguru import logger
from PIL import Image

# Initialize ZemServer for OCR
mcp = ZemServer("ocr")

def extract_pdf_pages(
    file_path: str,
    engine: str,
    ocr_engine,
    field: str,
    model_id: str | None,
    scanned_threshold: int = 50,
    zoom: float = 2.0,
    temp_dir: str = "/tmp",
):
    """Helper to process PDF pages with optional OCR for scanned content."""
    import fitz  # PyMuPDF
    
    results = []
    doc = fitz.open(file_path)
    
    # Ensure temp_dir exists
    os.makedirs(temp_dir, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Determine if we need to OCR (Strategy: text is too short or empty)
        is_scanned = len(text) < scanned_threshold
        
        if is_scanned:
            logger.info(f"Page {page_num + 1} appears scanned (text length: {len(text)}). Running OCR with {engine}...")
            # Render page to image for OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Temporary save for engine compatibility (engines expect path)
            temp_path = os.path.join(temp_dir, f"ocr_page_{os.getpid()}_{page_num}.png")
            img.save(temp_path)
            logger.debug(f"Saved temporary page image to: {temp_path}")
            
            try:
                ocr_result = ocr_engine.process(temp_path)
                final_text = ocr_result["text"]
                source = f"{engine}_ocr"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            final_text = text
            source = "digital_pdf"
            
        results.append(
            {
                "text": final_text,
                "engine": source,
                "metadata": {
                    "model_id": model_id,
                    "file_path": file_path,
                    "field": field,
                },
            }
        )
        
    doc.close()
    return results

@mcp.tool()
def extract_text(
    data: Any,
    field: str = "file_path",
    engine: str = "paddle_vl", 
    model_id: str = None,
    scanned_threshold: int = 50,
    zoom: float = 2.0,
    temp_dir: str = "/tmp"
) -> pd.DataFrame:
    """
    Extracts text from an image or PDF using the specified OCR engine.
    For PDFs, it will automatically handle scanned pages using the OCR engine.
    
    Args:
        field: Field name in each item that contains a file path or list of file paths.
        engine: The OCR engine to use ("tesseract", "paddle", "huggingface", "viet"). Defaults to "tesseract".
        model_id: Optional model ID for the 'huggingface' engine.
        scanned_threshold: Min characters required to skip OCR on PDF page. Defaults to 50.
        zoom: Rendering zoom factor for scanned PDF pages. Defaults to 2.0.
        temp_dir: Directory for temporary page images. Defaults to "/tmp".
    """
    items = mcp.get_data(data)
    items = items if isinstance(items, list) else [items]

    file_paths: List[str] = []
    for item in items:
        if not isinstance(item, dict) or field not in item:
            continue
        value = item[field]
        if isinstance(value, list):
            file_paths.extend([v for v in value if v])
        elif value:
            file_paths.append(value)

    if not file_paths:
        return mcp.save_output([])

    try:
        # Get engine from factory
        ocr_engine = OCREngineFactory.get_engine(engine, model_id=model_id)
        
        all_results = []

        for file_path in file_paths:
            logger.info(
                f"OCR Extraction: {file_path} using {engine} "
                f"(scanned_threshold={scanned_threshold}, zoom={zoom})"
            )

            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            # Handle PDF vs Image
            if str(file_path).lower().endswith(".pdf"):
                logger.info(f"Processing PDF file: {file_path}")
                page_results = extract_pdf_pages(
                    file_path,
                    engine,
                    ocr_engine,
                    field,
                    model_id,
                    scanned_threshold=scanned_threshold,
                    zoom=zoom,
                    temp_dir=temp_dir,
                )
                all_results.extend(page_results)
            else:
                # Process image
                result = ocr_engine.process(file_path)
                all_results.append({
                    "text": result.get("text", "") if isinstance(result, dict) else "",
                    "engine": result.get("engine", engine) if isinstance(result, dict) else engine,
                    "metadata": {
                        "model_id": model_id,
                        "file_path": file_path,
                        "field": field,
                    },
                })
        
        df = pd.DataFrame(all_results)
        logger.info(f"Output Dataframe: {df}")
        return mcp.save_output(df.to_dict(orient="records"))

        
    except Exception as e:
        logger.error(f"OCR Error with {engine}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"OCR failed: {str(e)}")

if __name__ == "__main__":
    mcp.run()
