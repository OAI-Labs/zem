import os
import pandas as pd
from xfmr_zem.server import ZemServer
from xfmr_zem.servers.ocr.engines import OCREngineFactory
from loguru import logger
from PIL import Image
import io

# Initialize ZemServer for OCR
mcp = ZemServer("ocr")

def extract_pdf_pages(file_path: str, engine: str, ocr_engine, model_id: str = None):
    """Helper to process PDF pages with optional OCR for scanned content."""
    import fitz  # PyMuPDF
    
    results = []
    doc = fitz.open(file_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Determine if we need to OCR (Strategy: text is too short or empty)
        is_scanned = len(text) < 50 
        
        if is_scanned:
            logger.info(f"Page {page_num + 1} appears scanned. Running OCR with {engine}...")
            # Render page to image for OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Temporary save for engine compatibility (engines expect path)
            temp_path = f"/tmp/ocr_page_{page_num}.png"
            img.save(temp_path)
            
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
            
        results.append({
            "text": final_text,
            "page": page_num + 1,
            "engine": source,
            "metadata": {"file": file_path, "is_scanned": is_scanned}
        })
        
    doc.close()
    return results

@mcp.tool()
async def extract_text(file_path: str, engine: str = "tesseract", model_id: str = None) -> pd.DataFrame:
    """
    Extracts text from an image or PDF using the specified OCR engine.
    For PDFs, it will automatically handle scanned pages using the OCR engine.
    
    Args:
        file_path: Path to the image or PDF file.
        engine: The OCR engine to use ("tesseract", "paddle", "huggingface", "viet"). Defaults to "tesseract".
        model_id: Optional model ID for the 'huggingface' engine.
    """
    logger.info(f"OCR Extraction: {file_path} using {engine}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Get engine from factory
        ocr_engine = OCREngineFactory.get_engine(engine, model_id=model_id)
        
        # Handle PDF vs Image
        if file_path.lower().endswith(".pdf"):
            logger.info(f"Processing PDF file: {file_path}")
            data = extract_pdf_pages(file_path, engine, ocr_engine, model_id)
            df = pd.DataFrame(data)
        else:
            # Process image
            result = ocr_engine.process(file_path)
            df = pd.DataFrame([{
                "text": result["text"],
                "engine": result["engine"],
                "metadata": result["metadata"]
            }])
        
        logger.info(f"Successfully extracted text from {file_path}")
        return df.to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"OCR Error with {engine}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"OCR failed: {str(e)}")

if __name__ == "__main__":
    mcp.run()
