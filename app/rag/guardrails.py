import re
from typing import List, Dict, Any
from app.config import Config

class CitationGuardrail:
    """
    Evaluates generated responses against retrieved context passages
    to verify source grounding, prevent hallucinations, and compute
    faithfulness metrics.
    """

    def __init__(self, min_confidence: float = Config.MIN_GROUNDING_CONFIDENCE):
        self.min_confidence = min_confidence
        self.sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def _tokenize_meaningful(self, text: str) -> set:
        """Extract alphanumeric words excluding stop words."""
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "have", "been",
            "will", "would", "could", "should", "your", "they", "them", "these",
            "those", "there", "their", "what", "which", "when", "where", "into"
        }
        return {w for w in words if w not in stopwords}

    def verify_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify factual claims in the answer against context passages.
        Returns detailed sentence-level audit and aggregate faithfulness score.
        """
        # Check for legitimate admissible refusals (model correctly states info is not specified)
        negative_indicators = [
            "not specified",
            "not mentioned",
            "no information",
            "cannot be answered",
            "not provided",
            "unavailable",
            "does not specify"
        ]
        if any(ind in answer.lower() for ind in negative_indicators):
            return {
                "grounding_score": 1.0,
                "grounding_percentage": 100.0,
                "is_grounded": True,
                "status": "VERIFIED_GROUNDED",
                "total_claims": 1,
                "grounded_claims": 1,
                "partial_claims": 0,
                "ungrounded_claims": 0,
                "flagged_claims": [],
                "sentence_audits": [{
                    "sentence": answer.strip(),
                    "verdict": "GROUNDED",
                    "confidence": 1.0,
                    "matched_terms": ["admissible_refusal"]
                }]
            }

        if not retrieved_chunks:
            return {
                "grounding_score": 0.0,
                "grounding_percentage": 0.0,
                "is_grounded": False,
                "status": "UNGROUNDED_NO_CONTEXT",
                "total_claims": 1,
                "grounded_claims": 0,
                "partial_claims": 0,
                "ungrounded_claims": 1,
                "sentence_audits": [],
                "flagged_claims": ["No context provided to ground the response."],
                "grounded_ratio": 0.0
            }

        # Combine all retrieved context text
        full_context_text = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()
        context_tokens = self._tokenize_meaningful(full_context_text)

        # Split answer into sentences
        sentences = [s.strip() for s in self.sentence_regex.split(answer) if len(s.strip()) > 10]
        if not sentences:
            return {
                "grounding_score": 1.0,
                "is_grounded": True,
                "status": "GROUNDED",
                "sentence_audits": [],
                "flagged_claims": [],
                "grounded_ratio": 1.0
            }

        sentence_audits = []
        grounded_count = 0
        partial_count = 0
        ungrounded_count = 0
        flagged_claims = []

        for sent in sentences:
            sent_tokens = self._tokenize_meaningful(sent)
            if not sent_tokens:
                continue

            # Calculate token intersection with retrieved context
            intersection = sent_tokens.intersection(context_tokens)
            overlap_ratio = len(intersection) / len(sent_tokens) if sent_tokens else 0.0

            # Substring heuristic for numbers, acronyms, or specific codes
            special_entities = re.findall(r"\b(?:\$?\d+(?:\.\d+)?%?|[A-Z]{2,})\b", sent)
            entity_matches = sum(1 for e in special_entities if e.lower() in full_context_text)
            entity_ratio = (entity_matches / len(special_entities)) if special_entities else 1.0

            # Combined confidence score
            confidence = (0.7 * overlap_ratio) + (0.3 * entity_ratio)

            if confidence >= 0.60:
                verdict = "GROUNDED"
                grounded_count += 1
            elif confidence >= 0.45:
                verdict = "PARTIAL"
                partial_count += 1
            else:
                verdict = "UNGROUNDED"
                ungrounded_count += 1
                flagged_claims.append(sent)

            sentence_audits.append({
                "sentence": sent,
                "verdict": verdict,
                "confidence": round(confidence, 3),
                "matched_terms": list(intersection)[:8]
            })

        total_sentences = len(sentence_audits)
        if total_sentences == 0:
            grounding_score = 1.0
        else:
            # Weighted average
            grounding_score = (grounded_count * 1.0 + partial_count * 0.6 + ungrounded_count * 0.1) / total_sentences

        grounding_score = round(grounding_score, 3)
        is_grounded = grounding_score >= self.min_confidence and len(flagged_claims) <= 1

        status = "VERIFIED_GROUNDED" if is_grounded else ("HALLUCINATION_RISK" if ungrounded_count > 0 else "PARTIALLY_GROUNDED")

        return {
            "grounding_score": grounding_score,
            "grounding_percentage": round(grounding_score * 100, 1),
            "is_grounded": is_grounded,
            "status": status,
            "total_claims": total_sentences,
            "grounded_claims": grounded_count,
            "partial_claims": partial_count,
            "ungrounded_claims": ungrounded_count,
            "flagged_claims": flagged_claims,
            "sentence_audits": sentence_audits
        }
