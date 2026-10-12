import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app import create_app
from app.config import Config
from app.retrieval.vector_store import VectorStore
from app.ingestion.queue_manager import QueueManager

def auto_seed_if_empty():
    """Automatically index sample PDFs if vector store has no documents."""
    vs = VectorStore.get_instance()
    if vs.get_total_chunks() == 0:
        print("[Auto-Seed] Initializing knowledge base with sample enterprise PDFs...")
        qm = QueueManager.get_instance()
        sample_docs = [
            (ROOT_DIR / "sample_docs" / "TechCorp_Employee_Handbook_2026.pdf", "TechCorp Global Employee Handbook (2026)"),
            (ROOT_DIR / "sample_docs" / "NovaCloud_Enterprise_Architecture_SLA.pdf", "NovaCloud Enterprise Architecture & SLA (2026)")
        ]
        for p, title in sample_docs:
            if p.exists():
                print(f"[Auto-Seed] Enqueueing: {p.name}")
                qm.submit_ingestion(str(p), custom_title=title)

if __name__ == "__main__":
    app = create_app()
    auto_seed_if_empty()
    
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Enterprise Document Intelligence & Advanced RAG System")
    print(f" Production Multi-Document Hybrid Q&A Engine")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
