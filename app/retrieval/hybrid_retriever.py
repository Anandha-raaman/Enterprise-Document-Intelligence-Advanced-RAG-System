from typing import List, Dict, Any, Optional
from app.config import Config
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever

class HybridRetriever:
    """
    Hybrid retriever combining dense semantic similarity and sparse BM25 keyword matching
    using Reciprocal Rank Fusion (RRF) and normalized weighted score fusion.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        alpha: float = Config.HYBRID_ALPHA,
        rrf_k: int = Config.RRF_K
    ):
        self.vector_store = vector_store or VectorStore.get_instance()
        self.bm25_retriever = bm25_retriever or BM25Retriever.get_instance()
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = Config.FUSED_TOP_K,
        mode: str = "rrf",  # 'rrf' or 'weighted'
        alpha: Optional[float] = None,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid search and return fused results alongside individual dense and sparse candidate pools.
        """
        effective_alpha = self.alpha if alpha is None else alpha

        # Step 1: Retrieve dense candidates from ChromaDB
        dense_results = self.vector_store.search(
            query=query,
            top_k=Config.DENSE_TOP_K,
            filter_meta=filter_meta
        )

        # Step 2: Retrieve sparse candidates from BM25
        sparse_results = self.bm25_retriever.search(
            query=query,
            top_k=Config.BM25_TOP_K
        )

        # Step 3: Candidate pool mapping
        candidates_map: Dict[str, Dict[str, Any]] = {}

        # Process dense candidates
        for item in dense_results:
            cid = item["chunk_id"]
            candidates_map[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_score": item.get("dense_score", 0.0),
                "dense_rank": item.get("dense_rank", 999),
                "bm25_score": 0.0,
                "bm25_rank": 999,
                "rrf_score": 0.0,
                "weighted_score": 0.0,
                "sources": ["dense"]
            }

        # Process sparse candidates
        for item in sparse_results:
            cid = item["chunk_id"]
            if cid in candidates_map:
                candidates_map[cid]["bm25_score"] = item.get("bm25_score", 0.0)
                candidates_map[cid]["bm25_rank"] = item.get("bm25_rank", 999)
                candidates_map[cid]["sources"].append("sparse")
            else:
                candidates_map[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_score": 0.0,
                    "dense_rank": 999,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "bm25_rank": item.get("bm25_rank", 999),
                    "rrf_score": 0.0,
                    "weighted_score": 0.0,
                    "sources": ["sparse"]
                }

        # Step 4: Calculate Fusion Scores
        for cid, cand in candidates_map.items():
            # RRF calculation
            rrf_dense = 1.0 / (self.rrf_k + cand["dense_rank"]) if cand["dense_rank"] < 900 else 0.0
            rrf_sparse = 1.0 / (self.rrf_k + cand["bm25_rank"]) if cand["bm25_rank"] < 900 else 0.0
            cand["rrf_score"] = rrf_dense + rrf_sparse

            # Weighted linear score
            cand["weighted_score"] = (effective_alpha * cand["dense_score"]) + ((1.0 - effective_alpha) * cand["bm25_score"])

        # Sort based on selected mode
        sort_key = "rrf_score" if mode == "rrf" else "weighted_score"
        fused_sorted = sorted(candidates_map.values(), key=lambda x: x[sort_key], reverse=True)

        for rank, item in enumerate(fused_sorted):
            item["fused_rank"] = rank + 1

        top_fused = fused_sorted[:top_k]

        return {
            "query": query,
            "fusion_mode": mode,
            "alpha": effective_alpha,
            "fused_results": top_fused,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "total_candidates": len(candidates_map)
        }
