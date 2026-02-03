import os
import pandas as pd
from xfmr_zem.server import ZemServer
from unstructured.partition.auto import partition
from loguru import logger

# Initialize ZemServer for Unstructured
mcp = ZemServer("unstructured")

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
