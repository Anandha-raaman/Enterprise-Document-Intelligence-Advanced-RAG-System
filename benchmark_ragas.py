import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import FlashRankReranker
from app.rag.generator import RAGGenerator
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.semantic_chunker import SemanticChunker

def ensure_indexed():
    """Ensure sample documents are in indexes before benchmark."""
    vs = VectorStore.get_instance()
    bm = BM25Retriever.get_instance()
    
    if vs.get_total_chunks() == 0:
        print("[Benchmark Setup] Indexing sample documents...")
        sample_files = [
            (ROOT_DIR / "sample_docs" / "TechCorp_Employee_Handbook_2026.pdf", "TechCorp Global Employee Handbook (2026)"),
            (ROOT_DIR / "sample_docs" / "NovaCloud_Enterprise_Architecture_SLA.pdf", "NovaCloud Enterprise Architecture & SLA (2026)")
        ]
        chunker = SemanticChunker()
        for p, title in sample_files:
            if p.exists():
                parser = PDFParser(str(p))
                parser.doc_title = title
                parsed = parser.parse()
                chunks = chunker.chunk_document(parsed)
                vs.add_chunks(chunks)
                bm.add_chunks(chunks)
                print(f"[Benchmark Setup] Indexed {len(chunks)} chunks from {p.name}")

def run_benchmark():
    ensure_indexed()

    vs = VectorStore.get_instance()
    bm = BM25Retriever.get_instance()
    hr = HybridRetriever(vs, bm)
    reranker = FlashRankReranker.get_instance()
    gen = RAGGenerator()
    evaluator = RagasEvaluator()

    print("\n" + "=" * 80)
    print(" RAGAS AUTOMATED EVALUATION BENCHMARK: ENTERPRISE MULTI-DOCUMENT RAG")
    print("=" * 80)
    print(" Benchmarking: Dense Vector vs Hybrid RRF vs FlashRank Cross-Encoder Reranking")
    print(" Evaluation Dimensions: Context Precision, Faithfulness, Answer Relevancy\n")

    t0 = time.time()
    report = evaluator.evaluate_pipeline(
        hybrid_retriever=hr,
        reranker=reranker,
        rag_generator=gen
    )
    elapsed = time.time() - t0

    metrics = report["summary_metrics"]
    cases = report["detailed_case_evaluations"]

    print(f"\n{'-'*80}")
    print(f" {'QUESTION & TARGET POLICY':<36} | {'DENSE':<6} | {'HYBRID':<6} | {'FLASHRANK':<10} | {'FAITH':<6} | {'RELEV':<6}")
    print(f"{'-'*80}")

    for c in cases:
        q_short = c["question"][:35]
        dp = c["context_precision_dense"]
        hp = c["context_precision_hybrid"]
        rp = c["context_precision_flashrank"]
        faith = c["faithfulness"]
        rel = c["answer_relevancy"]
        print(f" {q_short:<36} | {dp:<6.2f} | {hp:<6.2f} | {rp:<10.2f} | {faith:<6.2f} | {rel:<6.2f}")

    print(f"{'-'*80}\n")
    print(" SUMMARY METRICS:")
    print(f"  * Baseline Dense Vector Context Precision: {metrics['context_precision_dense']:.3f}")
    print(f"  * Hybrid (Dense + BM25) Context Precision:  {metrics['context_precision_hybrid']:.3f}")
    print(f"  * FlashRank Cross-Encoder Context Precision:{metrics['context_precision_flashrank']:.3f}")
    print(f"  * CONTEXT PRECISION LIFT:                  +{metrics['context_precision_lift_percent']}%")
    print(f"  * Retrieval Faithfulness (Zero-Halluc):    {metrics['faithfulness'] * 100:.1f}%")
    print(f"  * Answer Relevancy Score:                  {metrics['answer_relevancy'] * 100:.1f}%")
    print(f"  * Aggregate Ragas Composite Score:         {metrics['ragas_score_aggregate']:.3f}")
    print(f"  * Total Evaluation Runtime:                {elapsed:.2f}s\n")
    print("=" * 80 + "\n")

    # Save to data directory
    from app.config import DATA_DIR
    out_file = DATA_DIR / "ragas_benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Benchmark Report] Exported to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
