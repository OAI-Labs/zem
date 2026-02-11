import os
import pandas as pd
from xfmr_zem.server import ZemServer
from unstructured.partition.auto import partition
from loguru import logger
from typing import Any, Union, List, Dict

# Initialize ZemServer for Unstructured
mcp = ZemServer("unstructured")

@mcp.tool()
def extract_pdf_pages(
    data: Any,
    field: str = "file_path",
    zoom: float = 2.0,
) -> Any:
    """Extract page images from a PDF and return base64-encoded PNGs."""
    import pymupdf
    import base64

    # 1. Chuẩn hóa đầu vào thành list
    items = mcp.get_data(data)
    items = items if isinstance(items, list) else [items]

    all_results = []

    for item in items:
        # Kiểm tra tính hợp lệ của item
        if not isinstance(item, dict) or field not in item:
            logger.warning(f"Item does not contain field '{field}' or is not a dict.")
            continue

        file_path = item[field]
        
        # Kiểm tra file path
        if not file_path or not isinstance(file_path, str):
            continue

        try:
            # Mở PDF
            doc = pymupdf.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            continue

        # 2. Xử lý từng trang
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                img_bytes = pix.tobytes("png")
                base64_str = base64.b64encode(img_bytes).decode("utf-8")
                
                # 3. Tạo output item: Copy dữ liệu cũ -> Update dữ liệu mới
                # Điều này đảm bảo các trường dữ liệu khác trong 'item' được giữ nguyên
                page_result = item.copy()
                page_result.update({
                    "page_num": page_num + 1,
                    "image_base64": base64_str,
                })
                
                all_results.append(page_result)

            except Exception as e:
                logger.error(f"Error rendering page {page_num + 1} of {file_path}: {e}")

        doc.close()

    return mcp.save_output(all_results)


@mcp.tool()
async def parse_document(file_path: str, strategy: str = "fast") -> pd.DataFrame:
    """
    Parses a document (PDF, DOCX, HTML, etc.) and returns all text segments as a DataFrame.
    
    Args:
        file_path: Path to the document file.
        strategy: Partitioning strategy ("fast", "hi_res", "ocr_only"). Defaults to "fast".
    """
    logger.info(f"Parsing document: {file_path} with strategy: {strategy}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use unstructured to partition the file
    elements = partition(filename=file_path, strategy=strategy)
    
    # Convert elements to a list of dicts
    data = []
    for el in elements:
        data.append({
            "text": str(el),
            "type": el.category,
            "element_id": el.id,
            "metadata": el.metadata.to_dict() if hasattr(el, "metadata") else {}
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Extracted {len(df)} elements from {file_path}")
    return df

@mcp.tool()
async def extract_tables(file_path: str) -> pd.DataFrame:
    """
    Specifically extracts tables from a document and returns them.
    Note: Requires 'hi_res' strategy internally.
    """
    logger.info(f"Extracting tables from: {file_path}")
    
    # Partition with hi_res to get table structure
    elements = partition(filename=file_path, strategy="hi_res")
    
    # Filter for Table elements
    tables = [str(el) for el in elements if el.category == "Table"]
    
    if not tables:
        logger.warning(f"No tables found in {file_path}")
        return pd.DataFrame(columns=["table_content"])
        
    return pd.DataFrame({"table_content": tables})

if __name__ == "__main__":
    mcp.run()
