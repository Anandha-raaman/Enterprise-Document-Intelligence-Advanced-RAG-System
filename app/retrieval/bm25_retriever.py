import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.config import Config

class BM25Retriever:
    """
    Sparse keyword retriever using BM25Okapi with persistent serialization,
    tokenization, and score normalization.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, index_file: str = Config.BM25_INDEX_FILE):
        self.index_file = Path(index_file)
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load()

    def tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, extract alphanumeric terms."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        # Filter minimal common stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "over", "after", "is",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "this", "that", "these", "those", "it", "its"
        }
        return [t for t in tokens if t not in stopwords]

    def _save(self):
        """Serialize corpus and tokenized documents."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_chunks": self.corpus_chunks,
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(data, f)

    def _load(self):
        """Load corpus from disk if file exists."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_chunks = data.get("corpus_chunks", [])
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    if self.tokenized_corpus:
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
            except Exception as e:
                print(f"[BM25] Error loading index: {e}, resetting.")
                self.corpus_chunks = []
                self.tokenized_corpus = []
                self.bm25 = None

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add new chunks to the BM25 index."""
        if not chunks:
            return 0

        # Remove any existing chunks with the same chunk_ids to avoid duplicates
        incoming_ids = {c["chunk_id"] for c in chunks}
        filtered_corpus = []
        filtered_tokenized = []

        for c, tokens in zip(self.corpus_chunks, self.tokenized_corpus):
            if c["chunk_id"] not in incoming_ids:
                filtered_corpus.append(c)
                filtered_tokenized.append(tokens)

        # Tokenize incoming chunks
        for chunk in chunks:
            # Combine text and breadcrumb/section for richer lexical matching
            text_to_index = f"{chunk.get('doc_title', '')} {chunk.get('section', '')} {chunk['text']}"
            tokens = self.tokenize(text_to_index)
            filtered_corpus.append(chunk)
            filtered_tokenized.append(tokens)

        self.corpus_chunks = filtered_corpus
        self.tokenized_corpus = filtered_tokenized

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._save()

        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks associated with doc_id."""
        new_corpus = []
        new_tokenized = []
        for c, tokens in zip(self.corpus_chunks, self.tokenized_corpus):
            if c.get("doc_id") != doc_id:
                new_corpus.append(c)
                new_tokenized.append(tokens)

        self.corpus_chunks = new_corpus
        self.tokenized_corpus = new_tokenized
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
        self._save()
        return 1

    def search(self, query: str, top_k: int = Config.BM25_TOP_K) -> List[Dict[str, Any]]:
        """
        Execute BM25 keyword search.
        Returns top-K chunks with normalized scores in [0, 1].
        """
        if not self.bm25 or not self.corpus_chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        query_set = set(query_tokens)

        # Filter candidates that contain at least one query term
        matching_candidates = []
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            matches = query_set.intersection(set(doc_tokens))
            if matches:
                raw_score = float(doc_scores[idx])
                matching_candidates.append({
                    "idx": idx,
                    "matches_count": len(matches),
                    "raw_score": raw_score
                })

        if not matching_candidates:
            return []

        # Sort primarily by match count, then raw BM25 score
        matching_candidates.sort(key=lambda x: (x["matches_count"], x["raw_score"]), reverse=True)

        # Find min/max scores for normalization
        raw_values = [c["raw_score"] for c in matching_candidates]
        max_s = max(raw_values)
        min_s = min(raw_values)
        score_spread = max_s - min_s if max_s > min_s else 1.0

        candidates = []
        for rank, cand_info in enumerate(matching_candidates[:top_k]):
            idx = cand_info["idx"]
            raw_score = cand_info["raw_score"]
            chunk = self.corpus_chunks[idx]

            # Normalize to [0.3, 1.0] for matching items
            if max_s > min_s:
                norm_score = 0.3 + 0.7 * ((raw_score - min_s) / score_spread)
            else:
                norm_score = 1.0

            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk.get("doc_id", ""),
                    "doc_title": chunk.get("doc_title", ""),
                    "file_name": chunk.get("file_name", ""),
                    "doc_hash": chunk.get("doc_hash", ""),
                    "page_number": int(chunk.get("page_number", 1)),
                    "page_end": int(chunk.get("page_end", 1)),
                    "section": chunk.get("section", ""),
                    "subsection": chunk.get("subsection", ""),
                    "hierarchy_path": chunk.get("hierarchy_path", ""),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "token_count": int(chunk.get("token_count", 0)),
                    "char_count": int(chunk.get("char_count", len(chunk["text"])))
                },
                "bm25_score": round(norm_score, 4),
                "bm25_raw_score": raw_score,
                "bm25_rank": rank + 1,
                "source": "bm25_sparse"
            })

        return candidates
