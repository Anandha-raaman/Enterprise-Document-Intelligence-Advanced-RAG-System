import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from app.config import Config

class VectorStore:
    """
    ChromaDB wrapper providing persistent dense vector storage,
    MiniLM ONNX embeddings, metadata filtering, and similarity search.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.persist_dir = Config.CHROMA_PERSIST_DIR
        self.collection_name = Config.CHROMA_COLLECTION_NAME
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Use ONNX MiniLM embedding function (local, zero-API dependency)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index a batch of semantic chunks with rich metadata into ChromaDB.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            
            # ChromaDB metadata values must be str, int, float, or bool
            meta = {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "doc_id": str(chunk.get("doc_id", "")),
                "doc_title": str(chunk.get("doc_title", "")),
                "file_name": str(chunk.get("file_name", "")),
                "doc_hash": str(chunk.get("doc_hash", "")),
                "page_number": int(chunk.get("page_number", 1)),
                "page_end": int(chunk.get("page_end", 1)),
                "section": str(chunk.get("section", "")),
                "subsection": str(chunk.get("subsection", "")),
                "hierarchy_path": str(chunk.get("hierarchy_path", "")),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "token_count": int(chunk.get("token_count", 0)),
                "char_count": int(chunk.get("char_count", len(text)))
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)

        # Upsert in batches of 100 to avoid limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = Config.DENSE_TOP_K,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense semantic vector retrieval.
        Returns top-K chunks with normalized similarity scores.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        query_kwargs = {
            "query_texts": [query],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_meta:
            query_kwargs["where"] = filter_meta

        results = self.collection.query(**query_kwargs)
        
        candidates = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        for rank, (chunk_id, doc_text, meta, dist) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            # For cosine distance, similarity = 1 - distance (or max(0, 1 - dist))
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            candidates.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "metadata": meta,
                "dense_score": score,
                "dense_rank": rank + 1,
                "source": "dense_vector"
            })

        return candidates

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        self.collection.delete(where={"doc_id": doc_id})
        return 1

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks belonging to a specific document."""
        res = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        items = []
        if res and res["ids"]:
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                items.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta
                })
        return items

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregate all distinct documents currently in the vector store."""
        res = self.collection.get(include=["metadatas"])
        if not res or not res["metadatas"]:
            return []

        docs_map = {}
        for meta in res["metadatas"]:
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "doc_title": meta.get("doc_title", "Untitled"),
                    "file_name": meta.get("file_name", ""),
                    "doc_hash": meta.get("doc_hash", ""),
                    "chunk_count": 0,
                    "total_tokens": 0,
                    "max_page": 1,
                    "sections": set()
                }
            docs_map[doc_id]["chunk_count"] += 1
            docs_map[doc_id]["total_tokens"] += int(meta.get("token_count", 0))
            docs_map[doc_id]["max_page"] = max(docs_map[doc_id]["max_page"], int(meta.get("page_number", 1)))
            sec = meta.get("section")
            if sec and sec != "General":
                docs_map[doc_id]["sections"].add(sec)

        output = []
        for d in docs_map.values():
            d["sections"] = sorted(list(d["sections"]))[:10]
            output.append(d)
        return output

    def get_total_chunks(self) -> int:
        return self.collection.count()
