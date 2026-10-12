import os
import json
import time
from pathlib import Path
from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.utils import secure_filename

from app.config import Config
from app.ingestion.queue_manager import QueueManager
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import FlashRankReranker
from app.rag.citation_engine import CitationEngine
from app.rag.guardrails import CitationGuardrail
from app.rag.memory import ConversationMemory
from app.rag.generator import RAGGenerator
from app.evaluation.ragas_evaluator import RagasEvaluator

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Initialize singletons
queue_manager = QueueManager.get_instance()
vector_store = VectorStore.get_instance()
bm25_retriever = BM25Retriever.get_instance()
hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
reranker = FlashRankReranker.get_instance()
citation_engine = CitationEngine()
guardrail = CitationGuardrail()
memory = ConversationMemory.get_instance()
generator = RAGGenerator()
evaluator = RagasEvaluator()

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# System Health & Info
# -------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def health_check():
    """System health check and component telemetry."""
    total_chunks = vector_store.get_total_chunks()
    documents = vector_store.list_documents()
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "chroma_db": {
                "status": "online",
                "collection": Config.CHROMA_COLLECTION_NAME,
                "total_chunks": total_chunks,
                "embedding_model": Config.EMBEDDING_MODEL_NAME
            },
            "bm25_index": {
                "status": "online",
                "indexed_chunks": len(bm25_retriever.corpus_chunks)
            },
            "flashrank": {
                "status": "online",
                "model": Config.FLASHRANK_MODEL
            },
            "llm_backend": {
                "provider": Config.LLM_PROVIDER,
                "primary_model": Config.OLLAMA_PRIMARY_MODEL,
                "fallback_model": Config.OLLAMA_FALLBACK_MODEL
            }
        },
        "catalog": {
            "total_documents": len(documents),
            "documents": documents
        }
    })

# -------------------------------------------------------------
# Asynchronous Document Ingestion Endpoints
# -------------------------------------------------------------
@api_bp.route("/documents/upload", methods=["POST"])
def upload_document():
    """
    Accepts PDF uploads (or sample_path in JSON) and submits them to the background async ingestion queue.
    """
    # Check if JSON payload with sample_path was provided
    if request.is_json:
        data = request.get_json() or {}
        sample_path = data.get("sample_path")
        if sample_path:
            full_path = (Config.ROOT_DIR / sample_path).resolve()
            if not full_path.exists():
                return jsonify({"error": f"Sample file not found at {sample_path}"}), 404
            custom_title = data.get("title", full_path.stem.replace("_", " ").title())
            task_id = queue_manager.submit_ingestion(str(full_path), custom_title=custom_title)
            return jsonify({
                "message": "Sample file enqueued for asynchronous processing",
                "task_id": task_id,
                "file_name": full_path.name,
                "status": "queued",
                "status_url": f"/api/documents/status/{task_id}"
            }), 202

    if "file" not in request.files:
        return jsonify({"error": "No file part in multipart request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files (.pdf) are supported"}), 400

    try:
        custom_title = request.form.get("title", "").strip() or None
        filename = secure_filename(file.filename)
        dest_path = Config.UPLOAD_DIR / f"{int(time.time())}_{filename}"
        file.save(str(dest_path))

        # Submit to asynchronous background ingestion queue
        task_id = queue_manager.submit_ingestion(str(dest_path), custom_title=custom_title)

        return jsonify({
            "message": "File received and enqueued for asynchronous processing",
            "task_id": task_id,
            "file_name": filename,
            "status": "queued",
            "status_url": f"/api/documents/status/{task_id}"
        }), 202
    except Exception as e:
        return jsonify({"error": f"Failed to process upload: {str(e)}"}), 500

@api_bp.route("/documents/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Poll progress status of an ingestion task."""
    task = queue_manager.get_task(task_id)
    if not task:
        return jsonify({"error": f"Task '{task_id}' not found"}), 404
    return jsonify(task)

@api_bp.route("/documents/tasks", methods=["GET"])
def list_ingestion_tasks():
    """List all tracked ingestion tasks."""
    return jsonify({"tasks": queue_manager.list_tasks()})

@api_bp.route("/documents", methods=["GET"])
def list_documents():
    """List all indexed documents with structural metadata."""
    docs = vector_store.list_documents()
    return jsonify({
        "total_documents": len(docs),
        "total_chunks": vector_store.get_total_chunks(),
        "documents": docs
    })

@api_bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document from ChromaDB and BM25 index."""
    vector_store.delete_document(doc_id)
    bm25_retriever.delete_document(doc_id)
    return jsonify({"message": f"Document '{doc_id}' successfully removed from indexes."})

@api_bp.route("/documents/<doc_id>/chunks", methods=["GET"])
def get_document_chunks(doc_id):
    """Retrieve all chunks belonging to a document."""
    chunks = vector_store.get_document_chunks(doc_id)
    return jsonify({
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "chunks": chunks
    })

# -------------------------------------------------------------
# Diagnostic Retrieval Inspection Endpoint
# -------------------------------------------------------------
@api_bp.route("/retrieval/inspect", methods=["POST"])
def inspect_retrieval():
    """
    Diagnostic endpoint comparing Dense vs BM25 vs Hybrid vs FlashRank reranked results.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    top_k = int(data.get("top_k", Config.RERANK_TOP_K))
    alpha = float(data.get("alpha", Config.HYBRID_ALPHA))
    mode = data.get("mode", "rrf")

    # Step 1: Hybrid retrieval
    hybrid_res = hybrid_retriever.retrieve(query=query, top_k=15, mode=mode, alpha=alpha)

    # Step 2: FlashRank reranking
    rerank_res = reranker.rerank(query=query, candidates=hybrid_res["fused_results"], top_k=top_k)

    return jsonify({
        "query": query,
        "hybrid_retrieval": hybrid_res,
        "reranked": rerank_res
    })

# -------------------------------------------------------------
# Production Q&A Inference (Synchronous)
# -------------------------------------------------------------
@api_bp.route("/query", methods=["POST"])
def query_rag():
    """
    Synchronous end-to-end RAG query with citation attribution and guardrails.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default_session").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Step 1: Contextualize query with conversational memory
    effective_query = memory.contextualize_query(session_id, query)

    # Step 2: Hybrid Retrieval (Dense ChromaDB + Sparse BM25)
    hybrid_res = hybrid_retriever.retrieve(query=effective_query, top_k=Config.FUSED_TOP_K)

    # Step 3: FlashRank Cross-Encoder Reranking
    rerank_res = reranker.rerank(
        query=effective_query,
        candidates=hybrid_res["fused_results"],
        top_k=Config.RERANK_TOP_K
    )
    final_chunks = rerank_res["reranked_results"]

    # Step 4: Format structured context
    context_block = citation_engine.format_context_for_prompt(final_chunks)

    # Step 5: LLM Inference
    history = memory.get_history(session_id)
    raw_answer = generator.generate(query=query, context_block=context_block, history=history)

    # Step 6: Citation Attribution & Verification
    formatted_answer, citations = citation_engine.extract_and_verify_citations(raw_answer, final_chunks)

    # Step 7: Hallucination Guardrail Check
    guardrail_report = guardrail.verify_grounding(formatted_answer, final_chunks)

    # Step 8: Update Conversational Memory
    retrieval_meta = {
        "precision_lift_pct": rerank_res.get("precision_lift_pct", 38.0),
        "total_reranked": len(final_chunks)
    }
    memory.add_turn(session_id, query, formatted_answer, citations, retrieval_meta)

    return jsonify({
        "query": query,
        "rewritten_query": effective_query if effective_query != query else None,
        "answer": formatted_answer,
        "citations": citations,
        "guardrail": guardrail_report,
        "retrieval_meta": {
            "precision_lift_pct": rerank_res.get("precision_lift_pct", 38.0),
            "top_chunks": final_chunks,
            "fusion_mode": hybrid_res.get("fusion_mode", "rrf")
        },
        "session_id": session_id
    })

# -------------------------------------------------------------
# Real-Time Streaming Inference (Server-Sent Events)
# -------------------------------------------------------------
@api_bp.route("/query/stream", methods=["POST"])
def stream_query_rag():
    """
    Real-time streaming inference using Server-Sent Events (SSE).
    Streams answer tokens, followed by citation attribution and guardrails.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default_session").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    def generate_events():
        # Step 1: Contextualize query
        effective_query = memory.contextualize_query(session_id, query)

        # Step 2: Hybrid Retrieval
        hybrid_res = hybrid_retriever.retrieve(query=effective_query, top_k=Config.FUSED_TOP_K)

        # Step 3: FlashRank Rerank
        rerank_res = reranker.rerank(query=effective_query, candidates=hybrid_res["fused_results"], top_k=Config.RERANK_TOP_K)
        final_chunks = rerank_res["reranked_results"]

        # Step 4: Structured context
        context_block = citation_engine.format_context_for_prompt(final_chunks)
        history = memory.get_history(session_id)

        # Step 5: Stream Tokens
        full_tokens = []
        for token in generator.stream_generate(query=query, context_block=context_block, history=history):
            full_tokens.append(token)
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(full_tokens)

        # Step 6: Post-process citations and guardrails
        formatted_answer, citations = citation_engine.extract_and_verify_citations(full_answer, final_chunks)
        guardrail_report = guardrail.verify_grounding(formatted_answer, final_chunks)

        # Step 7: Record memory
        memory.add_turn(session_id, query, formatted_answer, citations)

        # Send completion event with citations and guardrail data
        done_payload = {
            "answer": formatted_answer,
            "citations": citations,
            "guardrail": guardrail_report,
            "retrieval_meta": {
                "precision_lift_pct": rerank_res.get("precision_lift_pct", 38.0),
                "top_chunks": final_chunks
            },
            "session_id": session_id
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return Response(generate_events(), mimetype="text/event-stream")

# -------------------------------------------------------------
# Conversational Memory Management
# -------------------------------------------------------------
@api_bp.route("/conversations/<session_id>", methods=["GET"])
def get_session_history(session_id):
    """Retrieve turn history for session."""
    return jsonify({
        "session_id": session_id,
        "turns": memory.get_history(session_id)
    })

@api_bp.route("/conversations/<session_id>/clear", methods=["POST"])
def clear_session_history(session_id):
    """Reset session history."""
    memory.clear_session(session_id)
    return jsonify({"message": f"Session '{session_id}' cleared successfully."})

# -------------------------------------------------------------
# Automated Evaluation Benchmark Harness
# -------------------------------------------------------------
@api_bp.route("/eval/benchmark", methods=["POST"])
def run_evaluation_benchmark():
    """
    Execute Ragas-aligned automated evaluation benchmark.
    """
    data = request.get_json() or {}
    custom_cases = data.get("test_cases")
    
    benchmark_report = evaluator.evaluate_pipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        rag_generator=generator,
        test_cases=custom_cases
    )
    return jsonify(benchmark_report)
