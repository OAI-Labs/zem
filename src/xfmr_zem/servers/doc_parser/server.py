"""
File: /app/xfmr-zem/src/xfmr_zem/servers/doc_parser/server.py
Description: OCR processing pipeline as a function.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

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

server = ZemServer("Document Parser", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def parse_document(
    files_to_process: Any,
    use_ocr: bool = True,
    use_corrector: bool = True,
    ocr_config: Dict[str, Any] = None,
    corrector_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1
) -> Any:
    """
    OCR pipeline processing function:
    - Initialize OCR/Corrector from config.
    - Process batch.
    - Return results (no file saving).
    
    Args:
        files_to_process (List[Path]): List of files to process.
        ocr_config (Dict[str, Any]): Configuration for OCR.
        use_ocr (bool): Whether to use OCR.
        use_corrector (bool): Whether to use Corrector.
        corrector_config (Optional[Dict[str, Any]]): Configuration for Corrector (if any).
        batch_size (int): Batch size.
        
    Returns:
        List[Dict[str, Any]]: List of processing results.
    """
    
    if not files_to_process:
        logger.warning("⚠️ No files provided to process.")
        return []

    # Initialize OCR
    
    params = server.parameters    

    ocr_config = params['parse_document'].get("ocr_config", None)
    corrector_config = params['parse_document'].get("corrector_config", None)

    use_ocr = params['parse_document'].get('use_ocr', True)
    use_corrector = params['parse_document'].get('use_corrector', False)
    batch_size = params['parse_document'].get('batch_size', 1)

    ocr_processor = None
    corrector_processor = None
    if use_ocr and ocr_config:
        logger.info("- Initializing OCR processor...")
        ocr_processor = ocr.make(ocr_config)
    
    if not ocr_processor:
        logger.warning("⚠️ OCR is disabled or not configured.")
        return []

    # Initialize Corrector
    if use_corrector and corrector_config:
        logger.info("- Initializing corrector processor...")
        corrector_processor = corrector.make(corrector_config)

    markdown_extractor = MarkdownExtractor()
    
    all_results = []
    total_files = len(files_to_process)

    for i in range(0, total_files, batch_size):
        batch_files = files_to_process[i : i + batch_size]
        logger.info(f"🚀 Processing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size} ({len(batch_files)} files)")

        # Delegate to OCR Processor
        results = ocr_processor.process_batch(batch_files)
        
        # Process results
        for result in results:
            markdown_content = result['markdown']
            
            # Apply Corrector
            if corrector_processor:
                logger.info("Apply Corrector to the parsed text")
                markdown_content = markdown_extractor.extract_and_correct(
                    markdown_content,
                    corrector_processor.correct
                )

            # Keep only the parsed text (parser output), remove file path and bounding boxes
            all_results.append({"markdown": markdown_content})
    return server.save_output(all_results)

if (__name__=="__main__"):
    server.run()
