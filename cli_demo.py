import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import FlashRankReranker
from app.rag.citation_engine import CitationEngine
from app.rag.guardrails import CitationGuardrail
from app.rag.generator import RAGGenerator
from app.rag.memory import ConversationMemory
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.semantic_chunker import SemanticChunker

def ensure_documents():
    vs = VectorStore.get_instance()
    bm = BM25Retriever.get_instance()
    if vs.get_total_chunks() == 0:
        print("\n[CLI] No indexed documents found. Indexing sample handbook...")
        handbook_path = ROOT_DIR / "sample_docs" / "TechCorp_Employee_Handbook_2026.pdf"
        sla_path = ROOT_DIR / "sample_docs" / "NovaCloud_Enterprise_Architecture_SLA.pdf"

        chunker = SemanticChunker()
        for p, title in [(handbook_path, "TechCorp Employee Handbook 2026"), (sla_path, "NovaCloud SLA 2026")]:
            if p.exists():
                parser = PDFParser(str(p))
                parser.doc_title = title
                parsed = parser.parse()
                chunks = chunker.chunk_document(parsed)
                vs.add_chunks(chunks)
                bm.add_chunks(chunks)
                print(f"[CLI] Indexed {len(chunks)} chunks from {p.name}")

def interactive_cli():
    ensure_documents()

    vs = VectorStore.get_instance()
    bm = BM25Retriever.get_instance()
    hybrid = HybridRetriever(vs, bm)
    reranker = FlashRankReranker.get_instance()
    citation_eng = CitationEngine()
    guardrails = CitationGuardrail()
    gen = RAGGenerator()
    mem = ConversationMemory.get_instance()
    session_id = "cli_session"

    print("\n" + "=" * 75)
    print(" Enterprise Document Intelligence & Advanced RAG System (CLI Demo)")
    print(" Type your questions, or type 'exit' or 'quit' to stop.")
    print("=" * 75)

    sample_questions = [
        "How many days of annual and sick leave do employees get?",
        "What is the daily per diem meal allowance and hotel cap for Tier 1 cities?",
        "What are the password requirements and remote access policies?",
        "What is NovaCloud's monthly uptime commitment and RTO for disaster recovery?"
    ]

    print("\nSample queries to try:")
    for idx, q in enumerate(sample_questions, 1):
        print(f" [{idx}] {q}")
    print("-" * 75 + "\n")

    while True:
        try:
            user_input = input("\nEnter Question > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting CLI demo. Goodbye!")
                break

            # Check if user entered a number 1-4
            if user_input.isdigit() and 1 <= int(user_input) <= len(sample_questions):
                user_input = sample_questions[int(user_input) - 1]
                print(f"Selected: {user_input}")

            print("\n[Stage 1 & 2] Running Hybrid Retrieval (Dense Vector + BM25)...")
            retrieval_res = hybrid.retrieve(query=user_input, top_k=8)

            print("[Stage 3] Executing FlashRank Cross-Encoder Reranking...")
            rerank_res = reranker.rerank(query=user_input, candidates=retrieval_res["fused_results"], top_k=4)
            top_chunks = rerank_res["reranked_results"]

            print("\n--- Reranked Candidate Ranking & Precision ---")
            for idx, c in enumerate(top_chunks, 1):
                delta = c.get("rank_delta", 0)
                delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "=")
                meta = c.get("metadata", {})
                print(f" Rank #{idx} (Shift: {delta_str}) | Score: {c['rerank_score']:.3f} | {meta.get('doc_title', '')} (p.{meta.get('page_number', 1)}) > {meta.get('section', '')}")

            # Structured Context
            context_block = citation_eng.format_context_for_prompt(top_chunks)

            print(f"\n[Stage 4] Generating Grounded Answer via {Config.OLLAMA_PRIMARY_MODEL}...")
            history = mem.get_history(session_id)
            raw_answer = gen.generate(query=user_input, context_block=context_block, history=history)

            # Attribution & Guardrails
            formatted_answer, citations = citation_eng.extract_and_verify_citations(raw_answer, top_chunks)
            guard_report = guardrails.verify_grounding(formatted_answer, top_chunks)

            mem.add_turn(session_id, user_input, formatted_answer, citations)

            print("\n" + "=" * 50)
            print("GROUNDED ANSWER:")
            print("=" * 50)
            print(formatted_answer)
            print("-" * 50)

            print("\nATTRIBUTED CITATIONS:")
            for cit in citations:
                print(f" [{cit['ref_id']}] {cit['doc_title']} (Page {cit['page_number']}, Section: {cit['section']})")

            print(f"\nHALLUCINATION GUARDRAIL AUDIT:")
            print(f" * Grounding Confidence:  {guard_report['grounding_percentage']}%")
            print(f" * Compliance Status:     {guard_report['status']}")
            print(f" * Grounded Sentences:    {guard_report['grounded_claims']}/{guard_report['total_claims']}")
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI demo. Goodbye!")
            break

if __name__ == "__main__":
    interactive_cli()
