"""
MinHash + LSH Implementation for Near-Duplicate Detection

Features:
- Uses datasketch library if available (faster)
- Falls back to pure Python implementation
- Supports Vietnamese tokenization for word-level shingles
- Diacritics normalization for OCR error handling
- Optional Redis backend for persistence
"""

import re
import hashlib
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from loguru import logger

from xfmr_zem.servers.deduplication.text_processing import remove_vietnamese_diacritics, tokenize_vietnamese


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
        """Compute MinHash signature using pure Python.
        
        Optimized: Pre-compute base MD5 hashes once, then apply permutations.
        This reduces MD5 computations from N×P to N (N=shingles, P=permutations).
        """
        if not shingles:
            return [0] * self.num_perm
        
        # Pre-compute base hashes for all shingles (expensive MD5 only once per shingle)
        base_hashes = [
            int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) 
            for s in shingles
        ]
        
        # Apply each permutation to pre-computed hashes
        signature = []
        for a, b, prime in self._hash_funcs:
            min_hash = min((a * h + b) % prime for h in base_hashes)
            signature.append(min_hash)
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
