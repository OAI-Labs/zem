"""
Vietnamese Text Processing Utilities

Provides:
- Diacritics normalization for OCR error handling
- Vietnamese tokenization using underthesea
"""

from typing import List
from loguru import logger

# =============================================================================
# VIETNAMESE DIACRITICS NORMALIZATION
# =============================================================================

VIETNAMESE_DIACRITICS_MAP = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
}


def remove_vietnamese_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics for normalization.
    This helps match documents with OCR errors (missing diacritics).
    
    Example:
        "Luật Doanh nghiệp" -> "luat doanh nghiep"
    """
    result = []
    for char in text.lower():
        if char in VIETNAMESE_DIACRITICS_MAP:
            result.append(VIETNAMESE_DIACRITICS_MAP[char])
        else:
            result.append(char)
    return ''.join(result)


# =============================================================================
# VIETNAMESE TOKENIZATION
# =============================================================================

# Vietnamese tokenizer (lazy loading)
_vi_tokenizer = None


def get_vi_tokenizer():
    """Lazy load Vietnamese tokenizer."""
    global _vi_tokenizer
    if _vi_tokenizer is None:
        try:
            from underthesea import word_tokenize
            _vi_tokenizer = word_tokenize
            logger.info("Vietnamese tokenizer (underthesea) loaded successfully")
        except ImportError:
            logger.warning("underthesea not found, falling back to simple whitespace tokenization")
            _vi_tokenizer = lambda text: text.split()
    return _vi_tokenizer


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenize Vietnamese text using underthesea.
    Returns list of tokens (compound words joined with underscore).
    
    Example:
        "học sinh giỏi" -> ["học_sinh", "giỏi"]
        "Luật Doanh nghiệp" -> ["Luật", "Doanh_nghiệp"]
    """
    tokenizer = get_vi_tokenizer()
    try:
        # underthesea returns list of tokens
        tokens = tokenizer(text)
        if isinstance(tokens, str):
            # Some versions return string with underscores
            tokens = tokens.split()
        return tokens
    except Exception as e:
        logger.warning(f"Tokenization failed, using whitespace: {e}")
        return text.split()
