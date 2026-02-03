import os
import sys
import re
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("data_juicer", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def clean_content(
    data: Any,
    remove_html: bool = True,
    remove_emojis: bool = False,
    text_column: str = "text"
) -> Any:
    """
    Flexible content cleaning tool using DataJuicer logic.
    """
    items = server.get_data(data)
    if not items: return []
    
    logger.info(f"DataJuicer: Cleaning content (remove_html={remove_html}, remove_emojis={remove_emojis})")
    
    for item in items:
        if text_column not in item: continue
        text = str(item[text_column])
        
        if remove_html:
            text = re.sub(r'<[^>]+>', '', text)
            
        if remove_emojis:
            # Targeted emoji removal: remove high-plane Unicode characters (likely emojis) 
            # while preserving Vietnamese diacritics and other standard Unicode text.
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            
        item[text_column] = text.strip()
        
    return server.save_output(items)

@server.tool()
def refining_filter(
    data: Any,
    min_len: int = 10,
    max_len: int = 100000,
    alphanumeric_ratio: float = 0.1,
    text_column: str = "text"
) -> Any:
    """
    Filter items based on technical refining metrics.
    """
    items = server.get_data(data)
    if not items: return []
    
    logger.info(f"DataJuicer: Refining filter (min_len={min_len}, alpha_ratio={alphanumeric_ratio})")
    
    filtered = []
    for item in items:
        text = str(item.get(text_column, ""))
        text_len = len(text)
        
        if not (min_len <= text_len <= max_len):
            continue
            
        # Alphanumeric ratio check
        alnum_count = len([c for c in text if c.isalnum()])
        if text_len > 0 and (alnum_count / text_len) < alphanumeric_ratio:
            continue
            
        filtered.append(item)
        
    return server.save_output(filtered)

@server.tool()
def language_id(
    data: Any,
    expected_lang: str = "vi",
    min_score: float = 0.8,
    text_column: str = "text"
) -> Any:
    """
    Heuristic language identification tool.
    """
    items = server.get_data(data)
    if not items: return []
    
    logger.info(f"DataJuicer: Filtering for language '{expected_lang}'")
    
    # Heuristic for Vietnamese
    vi_keywords = {'và', 'của', 'là', 'có', 'được', 'trong', 'cho', 'này', 'với', 'các'}
    
    filtered = []
    for item in items:
        text = str(item.get(text_column, "")).lower()
        words = set(text.split())
        
        if expected_lang == "vi":
            matches = len(words & vi_keywords)
            # Heuristic score: ratio of keywords found (simplified)
            if matches >= 2:
                filtered.append(item)
        else:
            # Fallback for other languages
            filtered.append(item)
            
    return server.save_output(filtered)

if __name__ == "__main__":
    server.run()
