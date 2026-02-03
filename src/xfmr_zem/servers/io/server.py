import os
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger
import sys

# Initialize server
server = ZemServer("io", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def load_jsonl(path: str, return_reference: bool = False) -> Any:
    """
    Load data from a JSONL file. If return_reference is True, returns metadata instead of data.
    """
    logger.info(f"IO: Loading JSONL from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    if return_reference:
        return {"path": path, "type": "jsonl", "size": os.path.getsize(path)}

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

@server.tool()
def save_jsonl(data: Any, path: str) -> Dict[str, Any]:
    """Save data (List or Path reference) to JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # If data is a reference (dict with path), we might want to move/copy it
    # but for now, assume data is a List if we are in save_jsonl
    if isinstance(data, dict) and "path" in data:
        import shutil
        shutil.copy(data["path"], path)
        return {"status": "copied", "path": path}

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"status": "success", "path": path, "count": len(data)}

@server.tool()
def load_parquet(path: str, return_reference: bool = True) -> Any:
    """
    Load data from a Parquet file. Defaults to return_reference=True for big data.
    """
    logger.info(f"IO: Handling Parquet from {path}")
    if not os.path.exists(path):
         raise FileNotFoundError(f"File not found: {path}")

    if return_reference:
        return {"path": path, "type": "parquet", "size": os.path.getsize(path)}
    
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

@server.tool()
def save_parquet(data: Any, path: str) -> Dict[str, Any]:
    """
    Save data to a Parquet file.
    """
    logger.info(f"IO: Saving Parquet to {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    if isinstance(data, dict) and "path" in data:
        src = data["path"]
        logger.info(f"IO: Copying reference from {src} to {path}")
        import shutil
        shutil.copy(src, path)
        return {"status": "copied", "path": path}
    
    logger.info("IO: Saving raw data to Parquet")
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    return {"status": "success", "path": path, "count": len(data)}

@server.tool()
def scan_directory(path: str, pattern: str = "*") -> List[Dict[str, Any]]:
    """
    Scan a directory and return a list of file references.
    Useful for parallelizing big data processing.
    """
    import glob
    logger.info(f"IO: Scanning directory {path} with pattern {pattern}")
    files = glob.glob(os.path.join(path, pattern))
    return [{"path": f, "type": "file_ref"} for f in files]

if __name__ == "__main__":
    server.run()
