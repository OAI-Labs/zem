import os
import re
import unicodedata
import sys
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Remove default handler and point to stderr
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize the server
server = ZemServer("data_juicer", parameter_file=os.path.join(os.path.dirname(__file__), "parameter.yaml"))

@server.tool()
def clean_html(data: Any) -> Any:
    """Remove HTML tags from text."""
    items = server.get_data(data)
    logger.info("DataJuicer: Cleaning HTML")
    result = []
    for item in items:
        text = str(item.get("text", ""))
        text = re.sub(r'<[^>]+>', '', text)
        new_item = item.copy()
        new_item["text"] = text
        result.append(new_item)
    
    # Support reference-based output for Big Data
    if server.parameters.get("return_reference", False):
        return server.save_output(result, format="parquet")
    return result

@server.tool()
def whitespace_normalization(data: Any) -> Any:
    """Normalize whitespace."""
    items = server.get_data(data)
    logger.info("DataJuicer: Whitespace Normalization")
    result = []
    for item in items:
        text = str(item.get("text", ""))
        text = re.sub(r'\s+', ' ', text).strip()
        new_item = item.copy()
        new_item["text"] = text
        result.append(new_item)

    if server.parameters.get("return_reference", False):
        return server.save_output(result, format="parquet")
    return result

@server.tool()
def text_length_filter(
    data: Any, 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None
) -> Any:
    """Filter documents based on text length."""
    items = server.get_data(data)
    params = server.parameters.get("text_length_filter", {})
    min_len = min_length if min_length is not None else params.get("min_length", 10)
    max_len = max_length if max_length is not None else params.get("max_length", 100000)
    
    logger.info(f"DataJuicer: Text Length Filter (min={min_len}, max={max_len})")
    result = [
        item for item in items 
        if min_len <= len(str(item.get("text", ""))) <= max_len
    ]

    if server.parameters.get("return_reference", False):
        return server.save_output(result, format="parquet")
    return result

@server.tool()
def language_filter(
    data: List[Dict[str, Any]], 
    lang: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter documents based on language heuristic.
    
    Args:
        data: List of dictionaries containing 'text' field.
        lang: Target language code ('en' or 'vi').
    """
    params = server.parameters.get("language_filter", {})
    target_lang = lang if lang is not None else params.get("lang", "en")
    
    logger.info(f"DataJuicer: Language Filter (target={target_lang})")
    
    # Simple heuristic from original processor
    if target_lang == 'en':
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
    elif target_lang == 'vi':
        common_words = {'và', 'của', 'là', 'có', 'được', 'trong', 'cho', 'này'}
    else:
        return data

    result = []
    for item in data:
        text = str(item.get("text", "")).lower()
        words = set(text.split())
        common_found = len(words & common_words)
        if common_found >= 2:
            result.append(item)
            
    logger.info(f"DataJuicer: Filtered {len(data)} -> {len(result)}")
    return result

@server.tool()
def document_simhash_dedup(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple deduplication based on prefix hashing.
    
    Args:
        data: List of dictionaries containing 'text' field.
    """
    logger.info("DataJuicer: SimHash Deduplication (Simple)")
    seen = set()
    result = []
    for item in data:
        text = str(item.get("text", ""))
        h = hash(text[:1000])  # Hash first 1000 chars as heuristic
        if h not in seen:
            seen.add(h)
            result.append(item)
            
    logger.info(f"DataJuicer: Dedup {len(data)} -> {len(result)}")
    return result

if __name__ == "__main__":
    server.run()
