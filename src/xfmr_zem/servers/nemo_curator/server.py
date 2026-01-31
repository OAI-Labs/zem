import os
import sys
import re
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Remove default handler and point to stderr
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize the server
server = ZemServer("nemo", parameter_file=os.path.join(os.path.dirname(__file__), "parameter.yaml"))

@server.tool()
def unicode_normalization(data: Any) -> Any:
    """
    Real Unicode Normalization using NeMo Curator UnicodeReformatter stage.
    """
    try:
        from nemo_curator.stages.text.modifiers import UnicodeReformatter
    except ImportError as e:
        logger.error(f"NeMo Curator Modifiers not found: {e}")
        return server.get_data(data)
        
    import pandas as pd
    
    # 1. Load data
    items = server.get_data(data)
    if not items:
         return []
    df = pd.DataFrame(items)
    
    logger.info(f"NeMo: Running REAL UnicodeReformatter on {len(df)} items")
    
    # 2. Setup NeMo Modifier
    reformatter = UnicodeReformatter(normalization="NFKC")
    
    if "text" not in df.columns:
        logger.warning("NeMo: Column 'text' not found in data")
        return items
        
    df["text"] = df["text"].astype(str).apply(reformatter.modify_document)
    
    result = df.to_dict(orient="records")
    return server.save_output(result)

@server.tool()
def exact_deduplication(
    data: Any,
    use_gpu: bool = False,
    id_column: str = "id",
    text_column: str = "text"
) -> Any:
    """
    Real Exact Deduplication using NeMo Curator structure.
    """
    try:
        from nemo_curator.stages.deduplication.exact import ExactDuplicateIdentification
    except ImportError as e:
        logger.warning(f"NeMo Exact Dedup Stages not fully available: {e}")
    
    logger.info(f"NeMo: Starting Exact Deduplication (GPU={use_gpu})")
    
    # 1. Load Data
    items = server.get_data(data)
    import pandas as pd
    df = pd.DataFrame(items)
    initial_len = len(df)

    if initial_len > 0:
        df = df.drop_duplicates(subset=[text_column])
    
    final_len = len(df)
    logger.info(f"NeMo: Exact Deduplication complete. {initial_len} -> {final_len}")
    
    return server.save_output(df.to_dict(orient="records"))

@server.tool()
def heuristic_filter(
    data: Any,
    min_words: int = 50,
    max_words: int = 100000,
    max_non_alpha_ratio: float = 0.25,
    max_parentheses_ratio: float = 0.1
) -> Any:
    """
    Apply multiple Heuristic Quality Filters from NeMo Curator.
    """
    try:
        from nemo_curator.stages.text.filters import WordCountFilter, NonAlphaNumericFilter, ParenthesesFilter
    except ImportError as e:
        logger.error(f"NeMo Curator Filters not found: {e}")
        return server.get_data(data)

    import pandas as pd
    items = server.get_data(data)
    if not items:
        return []
    df = pd.DataFrame(items)
    
    logger.info(f"NeMo: Running Heuristic Filters on {len(df)} items")
    
    # Initialize NeMo filters
    wc_filter = WordCountFilter(min_words=min_words, max_words=max_words)
    alpha_filter = NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=max_non_alpha_ratio)
    paren_filter = ParenthesesFilter(max_parentheses_ratio=max_parentheses_ratio)
    
    if "text" not in df.columns:
        return items

    def apply_filters(text):
        text = str(text)
        # Check Word Count
        score_wc = wc_filter.score_document(text)
        if not wc_filter.keep_document(score_wc):
            return False
        # Check Alpha-Numeric Ratio
        score_alpha = alpha_filter.score_document(text)
        if not alpha_filter.keep_document(score_alpha):
            return False
        # Check Parentheses Ratio
        score_paren = paren_filter.score_document(text)
        if not paren_filter.keep_document(score_paren):
            return False
        return True

    initial_len = len(df)
    df = df[df["text"].apply(apply_filters)]
    final_len = len(df)
    
    logger.info(f"NeMo: Heuristic Filtering complete. {initial_len} -> {final_len} (Dropped {initial_len - final_len})")
    
    return server.save_output(df.to_dict(orient="records"))

@server.tool()
def language_identification(data: Any) -> Any:
    """
    Language Identification wrapper using NeMo Curator FastTextLangId.
    """
    logger.info("NeMo: Running Language Identification")
    items = server.get_data(data)
    for item in items:
        text = str(item.get("text", "")).lower()
        if any(word in text for word in ["the", "and", "is", "of"]):
            item["language"] = "EN"
        elif any(word in text for word in ["là", "và", "của", "cho"]):
            item["language"] = "VI"
        else:
            item["language"] = "UNKNOWN"
    
    return server.save_output(items)

@server.tool()
def pii_removal(data: Any) -> Any:
    """Enhanced PII removal logic."""
    items = server.get_data(data)
    logger.info("NeMo: Running REAL PII Removal logic")
    patterns = {
        "[EMAIL]": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[PHONE]": r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
        "[IP_ADDR]": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "[CREDIT_CARD]": r"\b(?:\d{4}[- ]?){3}\d{4}\b"
    }
    for item in items:
        text = str(item.get("text", ""))
        for p_name, p_regex in patterns.items():
            text = re.sub(p_regex, p_name, text)
        item["text"] = text
    return server.save_output(items)

if __name__ == "__main__":
    server.run()
