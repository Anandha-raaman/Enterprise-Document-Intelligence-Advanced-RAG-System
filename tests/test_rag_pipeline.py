import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.semantic_chunker import SemanticChunker
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import FlashRankReranker
from app.rag.citation_engine import CitationEngine
from app.rag.guardrails import CitationGuardrail
from app.rag.memory import ConversationMemory
from app import create_app

class TestEnterpriseRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_pdf = ROOT_DIR / "sample_docs" / "TechCorp_Employee_Handbook_2026.pdf"
        assert cls.sample_pdf.exists(), "Sample PDF must exist for tests."
        
        # Parse and chunk document for tests
        parser = PDFParser(str(cls.sample_pdf))
        cls.parsed_doc = parser.parse()
        chunker = SemanticChunker(target_tokens=500, overlap_tokens=50)
        cls.chunks = chunker.chunk_document(cls.parsed_doc)

        # Index chunks so all retrieval and reranker tests have data
        VectorStore.get_instance().add_chunks(cls.chunks)
        BM25Retriever.get_instance().add_chunks(cls.chunks)

    def test_pdf_parsing_and_metadata(self):
        """Test PDF parser extracts structure, pages, and metadata."""
        self.assertEqual(self.parsed_doc["total_pages"], 2)
        self.assertTrue(len(self.parsed_doc["doc_hash"]) > 0)
        self.assertIn("Techcorp", self.parsed_doc["doc_title"].capitalize())
        
        # Verify section detection
        pages = self.parsed_doc["pages"]
        self.assertTrue(any("Leave" in p["active_section"] for p in pages))

    def test_semantic_chunker(self):
        """Test semantic chunking (~500 tokens, 10% overlap, no broken sentences)."""
        self.assertTrue(len(self.chunks) >= 2)
        
        for chunk in self.chunks:
            self.assertTrue(chunk["token_count"] > 20)
            self.assertIn("chunk_id", chunk)
            self.assertIn("hierarchy_path", chunk)
            self.assertIn("page_number", chunk)
            # Verify text doesn't end abruptly without punctuation
            self.assertTrue(chunk["text"].strip().endswith((".", "!", "?", "\"", "'", ")")))

    def test_bm25_retrieval(self):
        """Test sparse BM25 keyword matching."""
        bm25 = BM25Retriever.get_instance()
        bm25.add_chunks(self.chunks)

        results = bm25.search("gym membership reimbursement", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("gym", top["text"].lower())
        self.assertTrue(0.0 <= top["bm25_score"] <= 1.0)

    def test_vector_store_dense_retrieval(self):
        """Test dense vector similarity in ChromaDB."""
        vs = VectorStore.get_instance()
        vs.add_chunks(self.chunks)

        results = vs.search("annual vacation days leave policy", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("annual leave", top["text"].lower())
        self.assertTrue(0.0 <= top["dense_score"] <= 1.0)

    def test_hybrid_retriever_fusion(self):
        """Test hybrid fusion combining dense + sparse."""
        vs = VectorStore.get_instance()
        bm = BM25Retriever.get_instance()
        hybrid = HybridRetriever(vs, bm)

        res = hybrid.retrieve("sick leave physician medical certificate", top_k=5)
        fused = res["fused_results"]
        self.assertTrue(len(fused) > 0)
        top_cand = fused[0]
        self.assertIn("sick leave", top_cand["text"].lower())
        self.assertTrue("rrf_score" in top_cand)

    def test_flashrank_reranker(self):
        """Test FlashRank cross-encoder reranking."""
        reranker = FlashRankReranker.get_instance()
        vs = VectorStore.get_instance()
        bm = BM25Retriever.get_instance()
        hybrid = HybridRetriever(vs, bm)

        retrieval = hybrid.retrieve("multi-factor authentication remote VPN", top_k=6)
        rerank_res = reranker.rerank("multi-factor authentication remote VPN", retrieval["fused_results"], top_k=3)

        reranked = rerank_res["reranked_results"]
        self.assertTrue(len(reranked) > 0)
        top_reranked = reranked[0]
        self.assertIn("mfa", top_reranked["text"].lower())
        self.assertTrue(rerank_res["precision_lift_pct"] > 0)

    def test_citation_attribution_engine(self):
        """Test citation extraction, mapping, and formatting."""
        engine = CitationEngine()
        context_str = engine.format_context_for_prompt(self.chunks[:2])
        self.assertIn("[REF-1]", context_str)

        test_answer = "Full-time employees receive 20 days of annual leave [REF-1]."
        _, citations = engine.extract_and_verify_citations(test_answer, self.chunks[:2])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["ref_id"], "REF-1")

    def test_hallucination_guardrails(self):
        """Test source grounding verification and hallucination detection."""
        guard = CitationGuardrail()
        
        # Grounded claim
        grounded_ans = "Full-time employees receive 20 days of paid annual leave."
        report_grounded = guard.verify_grounding(grounded_ans, self.chunks)
        self.assertTrue(report_grounded["is_grounded"])
        self.assertTrue(report_grounded["grounding_score"] >= 0.70)

        # Fabricated claim (hallucination)
        hallucinated_ans = "Employees can adopt a pet llama and claim $50,000 in monthly cryptocurrency allowance."
        report_fake = guard.verify_grounding(hallucinated_ans, self.chunks)
        self.assertFalse(report_fake["is_grounded"])
        self.assertTrue(report_fake["ungrounded_claims"] > 0)

    def test_conversational_memory_contextualization(self):
        """Test multi-turn conversational query contextualization."""
        mem = ConversationMemory()
        sid = "test_sess"
        mem.add_turn(sid, "What are the rules for sick leave?", "Employees get 10 days of sick leave [REF-1].")

        # Follow-up with pronoun
        followup = "Can it be extended?"
        rewritten = mem.contextualize_query(sid, followup)
        self.assertIn("sick leave", rewritten.lower())

    def test_flask_api_routes(self):
        """Test Flask REST API endpoints."""
        app = create_app()
        client = app.test_client()

        # Health
        resp = client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")

        # Documents list
        resp = client.get("/api/documents")
        self.assertEqual(resp.status_code, 200)

        # Retrieval Inspection endpoint
        resp = client.post("/api/retrieval/inspect", json={"query": "annual leave policy", "top_k": 3})
        self.assertEqual(resp.status_code, 200)

if __name__ == "__main__":
    unittest.main()
