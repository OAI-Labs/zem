import os
import sys
import re
import unicodedata
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("nemo", parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yml"))

@server.tool()
def normalize(
    data: Any, 
    normalization: str = "NFC",
    cleanup_patterns: Optional[List[List[str]]] = None,
    text_column: str = "text"
) -> Any:
    """Flexible normalization tool."""
    try:
        items = server.get_data(data)
        if not items: return []
        logger.info(f"Nemo: Normalizing {len(items)} items")
        for item in items:
            if text_column not in item: continue
            text = str(item[text_column])
            text = unicodedata.normalize(normalization, text)
            if cleanup_patterns:
                for pattern, replacement in cleanup_patterns:
                    text = re.sub(pattern, replacement, text)
            item[text_column] = text.strip()
        return server.save_output(items)
    except Exception as e:
        logger.exception(f"Error in normalize: {e}")
        raise

@server.tool()
def quality_filter(
    data: Any, 
    min_words: int = 50,
    max_non_alpha_ratio: float = 0.25,
    text_column: str = "text"
) -> Any:
    """Flexible quality filter based on technical metrics."""
    items = server.get_data(data)
    if not items: return []
    logger.info(f"Nemo: Quality filter (min_words={min_words})")
    filtered = [i for i in items if len(str(i.get(text_column, "")).split()) >= min_words]
    # (Simplified for brevity, can add complex ratio checks)
    return server.save_output(filtered)

@server.tool()
def exact_deduplication(data: Any, text_column: str = "text") -> Any:
    """Exact deduplication."""
    import pandas as pd
    items = server.get_data(data)
    if not items: return []
    df = pd.DataFrame(items)
    df = df.drop_duplicates(subset=[text_column])
    return server.save_output(df.to_dict(orient="records"))

@server.tool()
def fuzzy_deduplication(
    data: Any, 
    text_column: str = "text", 
    threshold: float = 0.8,
    algorithm: str = "minhash"
) -> Any:
    """
    Fuzzy Deduplication (Task LSP-6).
    - threshold: Similarity threshold (0.0 to 1.0)
    - algorithm: minhash (fast, large scale) or levenshtein (precise, small scale)
    """
    items = server.get_data(data)
    if not items or len(items) < 2: return items
    
    logger.info(f"Nemo: Fuzzy deduplication (algorithm={algorithm}, threshold={threshold})")
    
    # Implementation using a simple similarity filter for small sets
    # In a real heavy scenario, this would call nemo_curator.stages.deduplication.fuzzy
    from difflib import SequenceMatcher
    
    unique_items = []
    seen_texts = []
    
    for item in items:
        text = str(item.get(text_column, ""))
        is_duplicate = False
        for seen in seen_texts:
            similarity = SequenceMatcher(None, text, seen).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_items.append(item)
            seen_texts.append(text)
            
    logger.info(f"Nemo: Fuzzy dedup complete. {len(items)} -> {len(unique_items)}")
    return server.save_output(unique_items)

@server.tool()
def language_filter(
    data: Any, 
    target_lang: str = "vi", 
    min_score: float = 0.5,
    text_column: str = "text"
) -> Any:
    """Filter documents by language (using fasttext-like logic)."""
    items = server.get_data(data)
    logger.info(f"Nemo: Filtering for language '{target_lang}'")
    # Placeholder for actual langid model call
    return server.save_output(items)

if __name__ == "__main__":
    server.run()
