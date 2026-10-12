import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.config import Config
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.semantic_chunker import SemanticChunker
from app.ingestion.metadata_extractor import MetadataExtractor
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever

class QueueManager:
    """
    Asynchronous background ingestion queue manager handling concurrent document
    parsing, semantic chunking, embedding generation, and hybrid index updates.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, max_workers: int = Config.MAX_INGESTION_WORKERS):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="IngestWorker")
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.vector_store = VectorStore.get_instance()
        self.bm25_retriever = BM25Retriever.get_instance()

    def submit_ingestion(self, file_path: str, custom_title: Optional[str] = None) -> str:
        """Enqueue document for background processing."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        path_obj = Path(file_path)

        task_record = {
            "task_id": task_id,
            "file_name": path_obj.name,
            "file_path": str(path_obj.resolve()),
            "status": "queued",
            "progress": 5,
            "stage": "Waiting in queue",
            "created_at": time.time(),
            "completed_at": None,
            "error": None,
            "doc_id": None,
            "doc_title": custom_title or path_obj.stem.replace("_", " ").title(),
            "stats": {}
        }

        with self.lock:
            self.tasks[task_id] = task_record

        self.executor.submit(self._worker_process, task_id, file_path, custom_title)
        return task_id

    def _worker_process(self, task_id: str, file_path: str, custom_title: Optional[str]):
        """Background thread worker pipeline."""
        try:
            self._update_task(task_id, status="parsing", progress=20, stage="Parsing PDF & extracting structure")
            
            # Step 1: Parse PDF
            parser = PDFParser(file_path)
            if custom_title:
                parser.doc_title = custom_title
            parsed_doc = parser.parse()
            doc_id = parsed_doc["doc_id"]
            doc_title = parsed_doc["doc_title"]

            self._update_task(task_id, status="chunking", progress=45, stage="Semantic chunking (~500 tokens, 10% overlap)")

            # Step 2: Semantic Chunking
            chunker = SemanticChunker(
                target_tokens=Config.TARGET_CHUNK_TOKENS,
                overlap_tokens=Config.CHUNK_OVERLAP_TOKENS
            )
            chunks = chunker.chunk_document(parsed_doc)

            if not chunks:
                raise ValueError("No extractable text or content found in document.")

            # Step 3: Domain Metadata & Tags
            summary = MetadataExtractor.extract_document_summary(parsed_doc)
            sample_text = " ".join([c["text"] for c in chunks[:5]])
            domain_tags = MetadataExtractor.extract_domain_tags(sample_text)

            self._update_task(task_id, status="indexing_dense", progress=70, stage="Generating dense embeddings (MiniLM/ONNX) & storing in ChromaDB")

            # Step 4: Index into ChromaDB
            dense_count = self.vector_store.add_chunks(chunks)

            self._update_task(task_id, status="indexing_sparse", progress=88, stage="Building BM25 sparse keyword index")

            # Step 5: Index into BM25
            sparse_count = self.bm25_retriever.add_chunks(chunks)

            # Step 6: Mark Complete
            total_tokens = sum(c.get("token_count", 0) for c in chunks)
            stats = {
                "total_pages": parsed_doc["total_pages"],
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "summary": summary,
                "domain_tags": domain_tags,
                "dense_indexed": dense_count,
                "sparse_indexed": sparse_count
            }

            self._update_task(
                task_id,
                status="completed",
                progress=100,
                stage="Ingestion completed successfully",
                completed_at=time.time(),
                doc_id=doc_id,
                doc_title=doc_title,
                stats=stats
            )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}: {traceback.format_exc()}"
            print(f"[Ingestion Error] {error_msg}")
            self._update_task(
                task_id,
                status="failed",
                progress=100,
                stage="Ingestion failed",
                completed_at=time.time(),
                error=str(e)
            )

    def _update_task(self, task_id: str, **kwargs):
        """Thread-safe update of task dictionary."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current status of a specific ingestion task."""
        with self.lock:
            return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tracked ingestion tasks, newest first."""
        with self.lock:
            return sorted(list(self.tasks.values()), key=lambda t: t.get("created_at", 0), reverse=True)
