"""
Deduplication Server - MinHash + LSH for Near-Duplicate Detection

This MCP server provides tools for detecting and removing near-duplicate
documents using MinHash signatures and Locality Sensitive Hashing (LSH).

Features:
- Two-Stage Entity-Aware Deduplication (filters template-based false positives)
- NER-based entity extraction using NlpHUST/ner-vietnamese-electra-base
- Vietnamese Diacritics Normalization (handles OCR errors)
- Support for Vietnamese word tokenization

Optimized for Vietnamese legal documents.
"""

import os
import sys
import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

from xfmr_zem.server import ZemServer
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# =============================================================================
# VIETNAMESE TEXT PROCESSING
# =============================================================================

# Vietnamese diacritics mapping for normalization
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


# =============================================================================
# NER-BASED ENTITY EXTRACTION FOR TWO-STAGE DEDUPLICATION
# =============================================================================

# NER model (lazy loading)
_ner_pipeline = None


def get_ner_pipeline():
    """
    Lazy load Vietnamese NER pipeline.
    Uses NlpHUST/ner-vietnamese-electra-base model.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logger.info("Loading Vietnamese NER model (NlpHUST/ner-vietnamese-electra-base)...")
            tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
            model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
            _ner_pipeline = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="simple",
                device=-1  # CPU, use 0 for GPU
            )
            logger.info("Vietnamese NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}. Falling back to regex-based extraction.")
            _ner_pipeline = None
    return _ner_pipeline


def extract_entities_ner(title: str) -> Set[str]:
    """
    Extract organization entities from title using NER.
    
    Args:
        title: Document title
    
    Returns:
        Set of organization names
    """
    ner = get_ner_pipeline()
    if ner is None:
        return set()
    
    try:
        results = ner(title)
        orgs = {r['word'] for r in results if r['entity_group'] == 'ORGANIZATION'}
        return orgs
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
        return set()


def extract_entities(doc: Dict) -> Dict[str, any]:
    """
    Extract key entities from a legal document for entity-aware deduplication.
    
    Uses NER model (NlpHUST/ner-vietnamese-electra-base) to extract organizations
    from document titles. This is more general and accurate than regex-based extraction.
    
    Args:
        doc: Document dictionary with 'title', 'markdown_content', 'issuing_body', etc.
    
    Returns:
        Dictionary of extracted entities including 'organizations' set
    """
    title = doc.get('title', '')
    
    entities = {
        'document_number': doc.get('document_number', ''),
        'issuing_body': doc.get('issuing_body', ''),
        'document_type': doc.get('document_type', ''),
        'issue_date': doc.get('issue_date', ''),
        'title': title,
    }
    
    # Extract organizations using NER
    entities['organizations'] = extract_entities_ner(title)
    
    # Fallback: Extract province from title or issuing_body
    province_match = re.search(
        r'(?:tỉnh|thành\s+phố)\s+([A-Za-zđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+(?:\s+[A-Za-zđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+)?)',
        title + ' ' + entities['issuing_body'],
        re.IGNORECASE
    )
    if province_match:
        entities['province'] = province_match.group(1).strip()
    
    return entities


def entities_match(e1: Dict, e2: Dict) -> Tuple[bool, str]:
    """
    Check if two documents have matching key entities using NER-extracted organizations.
    
    Returns:
        (is_match, reason): True if likely true duplicates, False if template-based
    """
    # PRIMARY CHECK: Compare NER-extracted organizations
    orgs1 = e1.get('organizations', set())
    orgs2 = e2.get('organizations', set())
    
    if orgs1 and orgs2:
        # If both have organizations extracted, they must match exactly
        if orgs1 != orgs2:
            return False, 'different_organizations'
    
    # SECONDARY CHECK: Check issuing body - different provinces issue similar docs
    ib1 = e1.get('issuing_body', '')
    ib2 = e2.get('issuing_body', '')
    
    if ib1 and ib2 and ib1 != ib2:
        return False, 'different_issuing_body'
    
    # FALLBACK: If no organizations extracted, use exact title match
    if not orgs1 and not orgs2:
        title1 = e1.get('title', '').strip().lower()
        title2 = e2.get('title', '').strip().lower()
        if title1 and title2 and title1 != title2:
            return False, 'different_titles'
    
    return True, 'match'


server = ZemServer(
    "deduplication",
    parameter_file=os.path.join(os.path.dirname(__file__), "parameter.yaml")
)


# =============================================================================
# CORE MINHASH + LSH IMPLEMENTATION
# =============================================================================

class MinHashLSH:
    """
    MinHash + LSH implementation for near-duplicate detection.
    
    Features:
    - Uses datasketch library if available (faster)
    - Falls back to pure Python implementation
    - Supports Vietnamese tokenization for word-level shingles
    - Diacritics normalization for OCR error handling
    - Optional Redis backend for persistence
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.5,
        num_bands: int = None,
        shingle_size: int = 5,
        shingle_type: str = "char",  # "char" or "word"
        use_vi_tokenizer: bool = True,  # Use Vietnamese tokenizer for word shingles
        normalize_diacritics: bool = True,  # Normalize Vietnamese diacritics
        redis_config: Dict = None  # Redis config for persistent storage
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.shingle_type = shingle_type
        self.use_vi_tokenizer = use_vi_tokenizer
        self.normalize_diacritics = normalize_diacritics
        self.redis_config = redis_config
        
        # Auto-calculate bands/rows for target threshold
        if num_bands is None:
            self.num_bands, self.rows_per_band = self._optimal_params(num_perm, threshold)
        else:
            self.num_bands = num_bands
            self.rows_per_band = num_perm // num_bands
        
        # Try to use datasketch for better performance
        self._use_datasketch = False
        self._use_redis = False
        
        try:
            from datasketch import MinHash, MinHashLSH as DatasketchLSH
            self._use_datasketch = True
            
            # Check for Redis backend
            if redis_config:
                try:
                    self._datasketch_lsh = DatasketchLSH(
                        threshold=threshold,
                        num_perm=num_perm,
                        storage_config={
                            'type': 'redis',
                            'redis': redis_config,
                            'basename': redis_config.get('basename', b'legal_dedup_lsh')
                        }
                    )
                    self._use_redis = True
                    logger.info(f"Using datasketch with Redis backend")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}, falling back to memory")
                    self._datasketch_lsh = DatasketchLSH(
                        threshold=threshold,
                        num_perm=num_perm
                    )
            else:
                self._datasketch_lsh = DatasketchLSH(
                    threshold=threshold,
                    num_perm=num_perm
                )
            logger.info(f"Using datasketch library for MinHash+LSH")
        except ImportError:
            logger.warning("datasketch not found, using pure Python implementation")
            self._hash_funcs = self._create_hash_functions(num_perm)
            self._buckets = defaultdict(list)  # band_id -> list of (doc_id, signature_band)
        
        self._signatures = {}  # doc_id -> signature
        self._doc_count = 0
    
    def _optimal_params(self, n: int, t: float) -> Tuple[int, int]:
        """
        Calculate optimal (bands, rows) for target threshold.
        P(candidate) ≈ 1 - (1 - t^r)^b where n = b * r
        """
        best_b, best_r = 1, n
        best_diff = float('inf')
        
        for b in range(1, n + 1):
            if n % b != 0:
                continue
            r = n // b
            # Approximate threshold for this configuration
            approx_t = (1 / b) ** (1 / r)
            diff = abs(approx_t - t)
            if diff < best_diff:
                best_diff = diff
                best_b, best_r = b, r
        
        logger.debug(f"LSH params: bands={best_b}, rows={best_r} for threshold={t}")
        return best_b, best_r
    
    def _create_hash_functions(self, n: int, prime: int = 4294967311) -> List:
        """Create n hash functions for MinHash."""
        import random
        random.seed(42)  # Reproducible
        hash_funcs = []
        for _ in range(n):
            a = random.randint(1, prime - 1)
            b = random.randint(0, prime - 1)
            hash_funcs.append((a, b, prime))
        return hash_funcs
    
    def _hash_shingle(self, shingle: str, a: int, b: int, prime: int) -> int:
        """Hash a shingle using linear hash function."""
        h = int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16)
        return (a * h + b) % prime
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Extract shingles from text."""
        text = self._preprocess_text(text)
        
        if self.shingle_type == "word":
            # Use Vietnamese tokenizer if enabled
            if self.use_vi_tokenizer:
                words = tokenize_vietnamese(text)
            else:
                words = text.split()
            
            if len(words) < self.shingle_size:
                return {" ".join(words)} if words else set()
            return {
                " ".join(words[i:i + self.shingle_size])
                for i in range(len(words) - self.shingle_size + 1)
            }
        else:  # char-level
            if len(text) < self.shingle_size:
                return {text} if text else set()
            return {
                text[i:i + self.shingle_size]
                for i in range(len(text) - self.shingle_size + 1)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for shingling."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Vietnamese diacritics if enabled
        if self.normalize_diacritics:
            text = remove_vietnamese_diacritics(text)
        
        return text
    
    def _compute_signature_pure(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature using pure Python."""
        if not shingles:
            return [0] * self.num_perm
        
        signature = []
        for a, b, prime in self._hash_funcs:
            min_hash = float('inf')
            for shingle in shingles:
                h = self._hash_shingle(shingle, a, b, prime)
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash if min_hash != float('inf') else 0)
        return signature
    
    def _compute_signature_datasketch(self, shingles: Set[str]):
        """Compute MinHash signature using datasketch."""
        from datasketch import MinHash
        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        return m
    
    def add_document(self, doc_id: str, text: str) -> bool:
        """Add a document to the LSH index. Returns False if doc_id already exists."""
        # Skip if already indexed
        if doc_id in self._signatures:
            logger.debug(f"Document {doc_id} already indexed, skipping")
            return False
        
        shingles = self._get_shingles(text)
        
        if self._use_datasketch:
            sig = self._compute_signature_datasketch(shingles)
            self._signatures[doc_id] = sig
            self._datasketch_lsh.insert(doc_id, sig, check_duplication=False)
        else:
            sig = self._compute_signature_pure(shingles)
            self._signatures[doc_id] = sig
            
            # Add to LSH buckets
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_sig = tuple(sig[start:end])
                bucket_key = (band_idx, band_sig)
                self._buckets[bucket_key].append(doc_id)
        
        self._doc_count += 1
        return True
    
    def query(self, doc_id: str) -> List[str]:
        """Find candidate duplicates for a document."""
        if doc_id not in self._signatures:
            return []
        
        if self._use_datasketch:
            # Filter out self from results
            results = self._datasketch_lsh.query(self._signatures[doc_id])
            return [r for r in results if r != doc_id]
        else:
            sig = self._signatures[doc_id]
            candidates = set()
            
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_sig = tuple(sig[start:end])
                bucket_key = (band_idx, band_sig)
                
                for candidate_id in self._buckets.get(bucket_key, []):
                    if candidate_id != doc_id:
                        candidates.add(candidate_id)
            
            return list(candidates)
    
    def estimate_jaccard(self, doc_id1: str, doc_id2: str) -> float:
        """Estimate Jaccard similarity between two documents."""
        if doc_id1 not in self._signatures or doc_id2 not in self._signatures:
            return 0.0
        
        if self._use_datasketch:
            return self._signatures[doc_id1].jaccard(self._signatures[doc_id2])
        else:
            sig1 = self._signatures[doc_id1]
            sig2 = self._signatures[doc_id2]
            matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
            return matches / len(sig1)
    
    def find_duplicates(self) -> List[Tuple[str, str, float]]:
        """Find all duplicate pairs above threshold."""
        duplicates = []
        seen_pairs = set()
        
        for doc_id in self._signatures:
            candidates = self.query(doc_id)
            for candidate_id in candidates:
                pair = tuple(sorted([doc_id, candidate_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    similarity = self.estimate_jaccard(doc_id, candidate_id)
                    if similarity >= self.threshold:
                        duplicates.append((doc_id, candidate_id, similarity))
        
        return sorted(duplicates, key=lambda x: -x[2])


# =============================================================================
# MCP TOOLS
# =============================================================================

@server.tool()
def minhash_dedup(
    data: Any,
    threshold: float = 0.9,
    num_perm: int = 128,
    shingle_size: int = 5,
    shingle_type: str = "char",
    use_vi_tokenizer: bool = False,
    normalize_diacritics: bool = True,
    text_column: str = "text",
    id_column: str = None,
    keep: str = "first",
    return_duplicates: bool = False
) -> Any:
    """
    Remove near-duplicate documents using MinHash + LSH.
    
    Args:
        data: Input data (list of dicts, file path, or reference)
        threshold: Jaccard similarity threshold (0.0-1.0). Default 0.9
        num_perm: Number of permutations for MinHash. Default 128
        shingle_size: Size of shingles. Default 5
        shingle_type: "char" for character-level, "word" for word-level
        use_vi_tokenizer: Use Vietnamese tokenizer (underthesea) for word shingles
        normalize_diacritics: Normalize Vietnamese diacritics (handles OCR errors). Default True
        text_column: Column containing text to compare
        id_column: Column to use as document ID (optional, uses index if not provided)
        keep: "first" to keep first occurrence, "longest" to keep longest text
        return_duplicates: If True, return duplicate pairs instead of deduplicated data
    
    Returns:
        Deduplicated data or list of duplicate pairs
    """
    items = server.get_data(data)
    if not items:
        return server.save_output([])
    
    logger.info(
        f"MinHash Dedup: {len(items)} documents, "
        f"threshold={threshold}, num_perm={num_perm}, "
        f"shingle_size={shingle_size}, shingle_type={shingle_type}, "
        f"normalize_diacritics={normalize_diacritics}"
    )
    
    # Initialize MinHash LSH
    lsh = MinHashLSH(
        num_perm=num_perm,
        threshold=threshold,
        shingle_size=shingle_size,
        shingle_type=shingle_type,
        use_vi_tokenizer=use_vi_tokenizer,
        normalize_diacritics=normalize_diacritics
    )
    
    # Build index
    doc_id_to_idx = {}
    for idx, item in enumerate(items):
        text = str(item.get(text_column, ""))
        if not text.strip():
            continue
        
        # Use provided ID column or generate one
        if id_column and id_column in item:
            doc_id = str(item[id_column])
        else:
            doc_id = f"doc_{idx}"
        
        doc_id_to_idx[doc_id] = idx
        lsh.add_document(doc_id, text)
    
    logger.info(f"Indexed {len(doc_id_to_idx)} documents")
    
    # Find duplicates
    duplicate_pairs = lsh.find_duplicates()
    logger.info(f"Found {len(duplicate_pairs)} duplicate pairs")
    
    if return_duplicates:
        # Return duplicate pairs with details
        result = []
        for doc_id1, doc_id2, similarity in duplicate_pairs:
            idx1 = doc_id_to_idx.get(doc_id1)
            idx2 = doc_id_to_idx.get(doc_id2)
            if idx1 is not None and idx2 is not None:
                result.append({
                    "doc1_id": doc_id1,
                    "doc2_id": doc_id2,
                    "doc1_idx": idx1,
                    "doc2_idx": idx2,
                    "similarity": round(similarity, 4),
                    "doc1_preview": str(items[idx1].get(text_column, ""))[:200],
                    "doc2_preview": str(items[idx2].get(text_column, ""))[:200]
                })
        return server.save_output(result)
    
    # Determine which documents to remove
    to_remove = set()
    
    for doc_id1, doc_id2, similarity in duplicate_pairs:
        idx1 = doc_id_to_idx.get(doc_id1)
        idx2 = doc_id_to_idx.get(doc_id2)
        
        if idx1 is None or idx2 is None:
            continue
        
        # Skip if one is already marked for removal
        if idx1 in to_remove or idx2 in to_remove:
            continue
        
        if keep == "first":
            # Keep the one with smaller index
            to_remove.add(max(idx1, idx2))
        elif keep == "longest":
            # Keep the one with longer text
            len1 = len(str(items[idx1].get(text_column, "")))
            len2 = len(str(items[idx2].get(text_column, "")))
            to_remove.add(idx1 if len1 < len2 else idx2)
        else:
            to_remove.add(idx2)
    
    # Filter out duplicates
    deduplicated = [
        item for idx, item in enumerate(items)
        if idx not in to_remove
    ]
    
    logger.info(
        f"Deduplication complete: {len(items)} -> {len(deduplicated)} "
        f"(removed {len(to_remove)} duplicates)"
    )
    
    return server.save_output(deduplicated)


@server.tool()
def compute_minhash_signatures(
    data: Any,
    num_perm: int = 128,
    shingle_size: int = 5,
    shingle_type: str = "char",
    use_vi_tokenizer: bool = True,
    text_column: str = "text",
    id_column: str = None
) -> Any:
    """
    Compute MinHash signatures for all documents without deduplication.
    Useful for incremental processing or custom similarity queries.
    
    Args:
        data: Input data
        num_perm: Number of permutations
        shingle_size: Size of shingles
        shingle_type: "char" or "word"
        use_vi_tokenizer: Use Vietnamese tokenizer for word shingles. Default True
        text_column: Column containing text
        id_column: Column to use as document ID
    
    Returns:
        List of documents with their MinHash signatures
    """
    items = server.get_data(data)
    if not items:
        return server.save_output([])
    
    logger.info(f"Computing MinHash signatures for {len(items)} documents")
    
    lsh = MinHashLSH(
        num_perm=num_perm,
        threshold=0.5,  # Not used for signature computation
        shingle_size=shingle_size,
        shingle_type=shingle_type,
        use_vi_tokenizer=use_vi_tokenizer
    )
    
    results = []
    for idx, item in enumerate(items):
        text = str(item.get(text_column, ""))
        
        if id_column and id_column in item:
            doc_id = str(item[id_column])
        else:
            doc_id = f"doc_{idx}"
        
        # Get shingles and compute signature
        shingles = lsh._get_shingles(text)
        
        if lsh._use_datasketch:
            sig = lsh._compute_signature_datasketch(shingles)
            # Convert to list for serialization
            signature = sig.hashvalues.tolist()
        else:
            signature = lsh._compute_signature_pure(shingles)
        
        result_item = item.copy()
        result_item["_minhash_signature"] = signature
        result_item["_doc_id"] = doc_id
        result_item["_num_shingles"] = len(shingles)
        results.append(result_item)
    
    logger.info(f"Computed signatures for {len(results)} documents")
    return server.save_output(results)


@server.tool()
def find_similar_documents(
    data: Any,
    query_text: str = None,
    query_idx: int = None,
    top_k: int = 10,
    threshold: float = 0.5,
    num_perm: int = 128,
    shingle_size: int = 5,
    shingle_type: str = "char",
    use_vi_tokenizer: bool = True,
    text_column: str = "text"
) -> Any:
    """
    Find documents similar to a query text or a document in the dataset.
    
    Args:
        data: Input data
        query_text: Text to find similar documents for
        query_idx: Index of document in data to use as query (alternative to query_text)
        top_k: Number of similar documents to return
        threshold: Minimum similarity threshold
        num_perm: Number of permutations
        shingle_size: Size of shingles
        shingle_type: "char" or "word"
        use_vi_tokenizer: Use Vietnamese tokenizer for word shingles. Default True
        text_column: Column containing text
    
    Returns:
        List of similar documents with similarity scores
    """
    items = server.get_data(data)
    if not items:
        return server.save_output([])
    
    # Determine query text
    if query_text is None:
        if query_idx is not None and 0 <= query_idx < len(items):
            query_text = str(items[query_idx].get(text_column, ""))
        else:
            logger.error("Either query_text or valid query_idx must be provided")
            return server.save_output([])
    
    logger.info(f"Finding similar documents (top_k={top_k}, threshold={threshold})")
    
    lsh = MinHashLSH(
        num_perm=num_perm,
        threshold=threshold,
        shingle_size=shingle_size,
        shingle_type=shingle_type,
        use_vi_tokenizer=use_vi_tokenizer
    )
    
    # Add query document
    lsh.add_document("__query__", query_text)
    
    # Add all other documents
    for idx, item in enumerate(items):
        if query_idx is not None and idx == query_idx:
            continue
        text = str(item.get(text_column, ""))
        if text.strip():
            lsh.add_document(f"doc_{idx}", text)
    
    # Find candidates
    candidates = lsh.query("__query__")
    
    # Compute exact similarities for candidates
    results = []
    for doc_id in candidates:
        idx = int(doc_id.split("_")[1])
        similarity = lsh.estimate_jaccard("__query__", doc_id)
        
        if similarity >= threshold:
            result_item = items[idx].copy()
            result_item["_similarity"] = round(similarity, 4)
            result_item["_doc_idx"] = idx
            results.append(result_item)
    
    # Sort by similarity and limit to top_k
    results = sorted(results, key=lambda x: -x["_similarity"])[:top_k]
    
    logger.info(f"Found {len(results)} similar documents")
    return server.save_output(results)


@server.tool()
def cluster_duplicates(
    data: Any,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_size: int = 5,
    shingle_type: str = "char",
    use_vi_tokenizer: bool = True,
    text_column: str = "text",
    id_column: str = None
) -> Any:
    """
    Cluster documents by similarity using Union-Find on duplicate pairs.
    
    Args:
        data: Input data
        threshold: Jaccard similarity threshold
        num_perm: Number of permutations
        shingle_size: Size of shingles
        shingle_type: "char" or "word"
        use_vi_tokenizer: Use Vietnamese tokenizer for word shingles. Default True
        text_column: Column containing text
        id_column: Column to use as document ID
    
    Returns:
        Documents with cluster assignments
    """
    items = server.get_data(data)
    if not items:
        return server.save_output([])
    
    logger.info(f"Clustering {len(items)} documents by similarity")
    
    lsh = MinHashLSH(
        num_perm=num_perm,
        threshold=threshold,
        shingle_size=shingle_size,
        shingle_type=shingle_type,
        use_vi_tokenizer=use_vi_tokenizer
    )
    
    # Build index
    doc_id_to_idx = {}
    for idx, item in enumerate(items):
        text = str(item.get(text_column, ""))
        if not text.strip():
            continue
        
        if id_column and id_column in item:
            doc_id = str(item[id_column])
        else:
            doc_id = f"doc_{idx}"
        
        doc_id_to_idx[doc_id] = idx
        lsh.add_document(doc_id, text)
    
    # Find duplicate pairs
    duplicate_pairs = lsh.find_duplicates()
    
    # Union-Find for clustering
    parent = {idx: idx for idx in range(len(items))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union duplicate pairs
    for doc_id1, doc_id2, _ in duplicate_pairs:
        idx1 = doc_id_to_idx.get(doc_id1)
        idx2 = doc_id_to_idx.get(doc_id2)
        if idx1 is not None and idx2 is not None:
            union(idx1, idx2)
    
    # Assign cluster IDs
    cluster_map = {}
    cluster_counter = 0
    
    results = []
    for idx, item in enumerate(items):
        result_item = item.copy()
        root = find(idx)
        
        if root not in cluster_map:
            cluster_map[root] = cluster_counter
            cluster_counter += 1
        
        result_item["_cluster_id"] = cluster_map[root]
        results.append(result_item)
    
    # Count cluster sizes
    cluster_sizes = defaultdict(int)
    for item in results:
        cluster_sizes[item["_cluster_id"]] += 1
    
    # Add cluster size to each item
    for item in results:
        item["_cluster_size"] = cluster_sizes[item["_cluster_id"]]
    
    num_clusters = len(cluster_map)
    num_singletons = sum(1 for s in cluster_sizes.values() if s == 1)
    
    logger.info(
        f"Clustering complete: {num_clusters} clusters, "
        f"{num_singletons} singletons, "
        f"{num_clusters - num_singletons} duplicate groups"
    )
    
    return server.save_output(results)


@server.tool()
def entity_aware_dedup(
    data: Any,
    threshold: float = 0.9,
    num_perm: int = 128,
    shingle_size: int = 5,
    shingle_type: str = "char",
    normalize_diacritics: bool = True,
    text_column: str = "markdown_content",
    id_column: str = "document_number",
    keep: str = "first",
    return_analysis: bool = False
) -> Any:
    """
    Two-Stage Entity-Aware Deduplication for legal documents.
    
    Stage 1: MinHash LSH finds candidate duplicate pairs (high recall)
    Stage 2: Entity comparison filters out template-based false positives
    
    This is optimized for Vietnamese legal documents where many documents
    share the same template but apply to different organizations.
    
    Args:
        data: Input data (list of dicts, file path, or reference)
        threshold: Jaccard similarity threshold for Stage 1. Default 0.9
        num_perm: Number of permutations for MinHash. Default 128
        shingle_size: Size of shingles. Default 5
        shingle_type: "char" for character-level, "word" for word-level
        normalize_diacritics: Normalize Vietnamese diacritics. Default True
        text_column: Column containing text to compare
        id_column: Column to use as document ID
        keep: "first" to keep first occurrence, "longest" to keep longest text
        return_analysis: If True, return detailed analysis instead of deduplicated data
    
    Returns:
        Deduplicated data or analysis results
    """
    items = server.get_data(data)
    if not items:
        return server.save_output([])
    
    logger.info(
        f"Two-Stage Entity-Aware Dedup: {len(items)} documents, "
        f"threshold={threshold}, normalize_diacritics={normalize_diacritics}"
    )
    
    # ========== STAGE 1: MinHash LSH ==========
    logger.info("Stage 1: Building MinHash LSH index...")
    
    lsh = MinHashLSH(
        num_perm=num_perm,
        threshold=threshold,
        shingle_size=shingle_size,
        shingle_type=shingle_type,
        use_vi_tokenizer=False,  # char-level is faster and good enough
        normalize_diacritics=normalize_diacritics
    )
    
    # Build index
    doc_id_to_idx = {}
    for idx, item in enumerate(items):
        text = str(item.get(text_column, ""))
        if not text.strip() or len(text) < 200:  # Skip very short docs
            continue
        
        if id_column and id_column in item:
            doc_id = f"{item[id_column]}_{idx}"  # Make unique
        else:
            doc_id = f"doc_{idx}"
        
        doc_id_to_idx[doc_id] = idx
        lsh.add_document(doc_id, text)
    
    logger.info(f"Indexed {len(doc_id_to_idx)} documents")
    
    # Find candidates
    stage1_candidates = lsh.find_duplicates()
    logger.info(f"Stage 1: Found {len(stage1_candidates)} candidate pairs")
    
    # ========== STAGE 2: Entity Comparison ==========
    logger.info("Stage 2: Entity-aware filtering...")
    
    true_duplicates = []
    false_positives = []
    
    for doc_id1, doc_id2, similarity in stage1_candidates:
        idx1 = doc_id_to_idx.get(doc_id1)
        idx2 = doc_id_to_idx.get(doc_id2)
        
        if idx1 is None or idx2 is None:
            continue
        
        # Extract entities
        e1 = extract_entities(items[idx1])
        e2 = extract_entities(items[idx2])
        
        # Compare entities
        is_match, reason = entities_match(e1, e2)
        
        if is_match:
            true_duplicates.append((doc_id1, doc_id2, similarity, idx1, idx2))
        else:
            false_positives.append((doc_id1, doc_id2, similarity, reason, idx1, idx2))
    
    logger.info(
        f"Stage 2: {len(true_duplicates)} true duplicates, "
        f"{len(false_positives)} false positives filtered"
    )
    
    if return_analysis:
        # Return detailed analysis as list of records (for parquet compatibility)
        # First record contains summary stats
        fp_reasons = defaultdict(int)
        for _, _, _, reason, _, _ in false_positives:
            fp_reasons[reason] += 1
        
        analysis_records = []
        
        # Summary record
        analysis_records.append({
            "type": "summary",
            "total_documents": len(items),
            "indexed_documents": len(doc_id_to_idx),
            "stage1_candidates": len(stage1_candidates),
            "true_duplicates": len(true_duplicates),
            "false_positives_filtered": len(false_positives),
            "reduction_rate": round(len(false_positives) / max(1, len(stage1_candidates)), 4),
            "fp_different_organizations": fp_reasons.get("different_organizations", 0),
            "fp_different_issuing_body": fp_reasons.get("different_issuing_body", 0),
            "fp_different_titles": fp_reasons.get("different_titles", 0),
            "doc1_id": None,
            "doc2_id": None,
            "similarity": None,
            "doc1_title": None,
            "doc2_title": None,
        })
        
        # True duplicate pairs
        for d1, d2, sim, i1, i2 in true_duplicates[:50]:
            analysis_records.append({
                "type": "true_duplicate",
                "total_documents": None,
                "indexed_documents": None,
                "stage1_candidates": None,
                "true_duplicates": None,
                "false_positives_filtered": None,
                "reduction_rate": None,
                "fp_different_organizations": None,
                "fp_different_issuing_body": None,
                "fp_different_titles": None,
                "doc1_id": d1.rsplit('_', 1)[0],
                "doc2_id": d2.rsplit('_', 1)[0],
                "similarity": round(sim, 4),
                "doc1_title": items[i1].get('title', '')[:100],
                "doc2_title": items[i2].get('title', '')[:100],
            })
        
        return server.save_output(analysis_records)
    
    # Determine which documents to remove
    to_remove = set()
    
    for doc_id1, doc_id2, similarity, idx1, idx2 in true_duplicates:
        # Skip if one is already marked for removal
        if idx1 in to_remove or idx2 in to_remove:
            continue
        
        if keep == "first":
            to_remove.add(max(idx1, idx2))
        elif keep == "longest":
            len1 = len(str(items[idx1].get(text_column, "")))
            len2 = len(str(items[idx2].get(text_column, "")))
            to_remove.add(idx1 if len1 < len2 else idx2)
        else:
            to_remove.add(idx2)
    
    # Filter out duplicates
    deduplicated = [
        item for idx, item in enumerate(items)
        if idx not in to_remove
    ]
    
    logger.info(
        f"Entity-Aware Dedup complete: {len(items)} -> {len(deduplicated)} "
        f"(removed {len(to_remove)} true duplicates)"
    )
    
    return server.save_output(deduplicated)


if __name__ == "__main__":
    server.run()
