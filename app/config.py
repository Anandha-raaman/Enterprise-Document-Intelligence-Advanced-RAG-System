import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_DIR = DATA_DIR / "bm25_index"
SAMPLE_DOCS_DIR = ROOT_DIR / "sample_docs"

# Ensure directories exist
for path in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR, BM25_DIR, SAMPLE_DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

class Config:
    # Base Directories
    ROOT_DIR = ROOT_DIR
    APP_DIR = APP_DIR
    DATA_DIR = DATA_DIR
    UPLOAD_DIR = UPLOAD_DIR
    CHROMA_DIR = CHROMA_DIR
    BM25_DIR = BM25_DIR
    SAMPLE_DOCS_DIR = SAMPLE_DOCS_DIR

    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "enterprise-rag-secret-token-2026")
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1")

    # Ingestion & Semantic Chunking Settings
    # 500 tokens target with 10% overlap (~50 tokens)
    TARGET_CHUNK_TOKENS = 500
    CHUNK_OVERLAP_TOKENS = 50
    MIN_CHUNK_TOKENS = 40
    ESTIMATED_CHARS_PER_TOKEN = 4  # heuristic for character-token estimation

    # ChromaDB & Vector Store Settings
    CHROMA_PERSIST_DIR = str(CHROMA_DIR)
    CHROMA_COLLECTION_NAME = "enterprise_rag_knowledge"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # BM25 Sparse Index
    BM25_INDEX_FILE = str(BM25_DIR / "bm25_corpus.pkl")

    # Hybrid Retrieval Settings
    DENSE_TOP_K = 15
    BM25_TOP_K = 15
    FUSED_TOP_K = 15
    HYBRID_ALPHA = 0.5  # 0.5 balanced, 1.0 dense only, 0.0 sparse only
    RRF_K = 60  # Reciprocal Rank Fusion constant

    # FlashRank Cross-Encoder Reranker Settings
    FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
    RERANK_TOP_K = 3
    RELEVANCE_SCORE_THRESHOLD = 0.005  # Minimum cross-encoder confidence to consider chunk relevant

    # LLM Settings
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # 'ollama', 'gemini', 'mock'
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_PRIMARY_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "gemma3:270m")
    LLM_TEMPERATURE = 0.0
    LLM_MAX_TOKENS = 250
    LLM_REQUEST_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

    # Guardrails & Citations
    STRICT_CITATIONS = True
    HALLUCINATION_THRESHOLD = 0.60
    MIN_GROUNDING_CONFIDENCE = 0.70

    # Asynchronous Ingestion Queue
    MAX_INGESTION_WORKERS = 2
