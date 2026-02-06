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
from typing import Any
from collections import defaultdict

from xfmr_zem.server import ZemServer
from loguru import logger

# Import from local modules
from xfmr_zem.servers.deduplication.minhash import MinHashLSH
from xfmr_zem.servers.deduplication.ner import extract_entities, entities_match

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize server
server = ZemServer(
    "deduplication",
    parameter_file=os.path.join(os.path.dirname(__file__), "parameters.yaml")
)


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
    
    # Union-Find for clustering (iterative implementation to avoid recursion limit)
    parent = {idx: idx for idx in range(len(items))}
    
    def find(x):
        """Find root with path compression (iterative to avoid RecursionError)."""
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_node = parent[x]
            parent[x] = root
            x = next_node
        return root
    
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
    
    # Cache extracted entities to avoid redundant NER calls
    # (a document may appear in multiple candidate pairs)
    entity_cache = {}
    
    for doc_id1, doc_id2, similarity in stage1_candidates:
        idx1 = doc_id_to_idx.get(doc_id1)
        idx2 = doc_id_to_idx.get(doc_id2)
        
        if idx1 is None or idx2 is None:
            continue
        
        # Extract entities with caching
        if idx1 not in entity_cache:
            entity_cache[idx1] = extract_entities(items[idx1])
        if idx2 not in entity_cache:
            entity_cache[idx2] = extract_entities(items[idx2])
        
        e1 = entity_cache[idx1]
        e2 = entity_cache[idx2]
        
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
        
        # True duplicate pairs (save ALL, not just first 50)
        for d1, d2, sim, i1, i2 in true_duplicates:
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
