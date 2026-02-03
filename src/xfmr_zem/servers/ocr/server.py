import pandas as pd
from xfmr_zem.server import ZemServer
from xfmr_zem.servers.ocr.engines import OCREngineFactory
from loguru import logger

# Initialize ZemServer for OCR
mcp = ZemServer("ocr")

@mcp.tool()
async def extract_text(file_path: str, engine: str = None, model_id: str = None) -> pd.DataFrame:
    """
    Extracts text from an image using the specified OCR engine.
    
    Args:
        file_path: Path to the image file.
        engine: The OCR engine to use ("tesseract", "paddle", "huggingface", "viet"). Defaults to "tesseract".
        model_id: Optional model ID for the 'huggingface' engine (e.g., "Qwen/Qwen2-VL-2B-Instruct").
    """
    logger.info(f"OCR Extraction: {file_path} using {engine} (model: {model_id})")
    
    try:
        # Get engine from factory (SOLID Strategy Pattern)
        ocr_engine = OCREngineFactory.get_engine(engine, model_id=model_id)
        
        # Process image
        result = ocr_engine.process(file_path)
        
        # Structure as a single-row DataFrame for Zem compatibility
        # We wrap in a list to ensure pandas creates a row
        df = pd.DataFrame([{
            "text": result["text"],
            "engine": result["engine"],
            "metadata": result["metadata"]
        }])
        
        logger.info(f"Successfully extracted text using {engine}")
        return df.to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"OCR Error with {engine}: {e}")
        raise RuntimeError(f"OCR failed: {str(e)}")

if __name__ == "__main__":
    mcp.run()
