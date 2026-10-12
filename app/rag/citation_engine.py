import re
from typing import List, Dict, Any, Tuple

class CitationEngine:
    """
    Formats retrieval context with strict numbered reference tags and extracts,
    validates, and anchors citations in LLM responses.
    """

    @staticmethod
    def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
        """
        Produce structured reference block for system prompt.
        Format:
        [REF-1] Document: "TechCorp Employee Handbook" | Page: 1 | Section: "Leave Policy"
        Content: ...
        """
        if not chunks:
            return "No relevant context found."

        context_blocks = []
        for idx, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            doc_title = meta.get("doc_title", "Document")
            page_num = meta.get("page_number", 1)
            section = meta.get("section", "General")
            text = c.get("text", "").replace("\x7f", "- ").strip()

            block = (
                f"[REF-{idx}] {doc_title} (Page {page_num}, Section: {section}):\n"
                f"{text}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)

    @staticmethod
    def extract_and_verify_citations(
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Find citations such as [REF-1], [Doc: X, Page: Y], or [Ref 1] in text,
        link them to retrieved chunk metadata, and generate rich citation cards.
        """
        citations = []
        seen_refs = set()

        # Build lookup table for [REF-i]
        ref_lookup = {}
        for idx, c in enumerate(retrieved_chunks, 1):
            ref_lookup[f"REF-{idx}"] = (idx, c)
            ref_lookup[f"REF {idx}"] = (idx, c)
            ref_lookup[f"{idx}"] = (idx, c)

        # Pattern 1: [REF-1] or [Ref 1] or [1]
        ref_pattern = re.compile(r"\[(?:REF-?|Ref\s*)?(\d+)\]", re.IGNORECASE)
        matches = ref_pattern.finditer(answer_text)

        for match in matches:
            ref_num_str = match.group(1)
            key = f"REF-{ref_num_str}"
            if key in ref_lookup and key not in seen_refs:
                seen_refs.add(key)
                idx, chunk = ref_lookup[key]
                meta = chunk.get("metadata", {})
                citations.append({
                    "ref_id": f"REF-{idx}",
                    "ref_index": idx,
                    "doc_title": meta.get("doc_title", "Document"),
                    "file_name": meta.get("file_name", ""),
                    "page_number": meta.get("page_number", 1),
                    "section": meta.get("section", "General"),
                    "hierarchy_path": meta.get("hierarchy_path", ""),
                    "excerpt": chunk.get("text", "")[:350] + ("..." if len(chunk.get("text", "")) > 350 else ""),
                    "rerank_score": round(chunk.get("rerank_score", 0.0), 3)
                })

        # Pattern 2: [Doc: ..., Page: ..., Section: ...]
        doc_pattern = re.compile(r"\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)(?:,\s*Section:\s*([^\]]+))?\]", re.IGNORECASE)
        for match in doc_pattern.finditer(answer_text):
            doc_name = match.group(1).strip()
            page_no = int(match.group(2).strip())
            sec_name = match.group(3).strip() if match.group(3) else "General"
            
            # Check if this matches an existing chunk
            found_chunk = None
            for idx, c in enumerate(retrieved_chunks, 1):
                m = c.get("metadata", {})
                if str(m.get("page_number")) == str(page_no):
                    found_chunk = (idx, c)
                    break
            
            ref_key = f"DOC-{doc_name}-P{page_no}"
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                if found_chunk:
                    idx, c = found_chunk
                    meta = c.get("metadata", {})
                    citations.append({
                        "ref_id": f"REF-{idx}",
                        "ref_index": idx,
                        "doc_title": meta.get("doc_title", doc_name),
                        "file_name": meta.get("file_name", ""),
                        "page_number": page_no,
                        "section": sec_name or meta.get("section", "General"),
                        "hierarchy_path": meta.get("hierarchy_path", f"{doc_name} > {sec_name}"),
                        "excerpt": c.get("text", "")[:350],
                        "rerank_score": round(c.get("rerank_score", 0.0), 3)
                    })
                else:
                    citations.append({
                        "ref_id": f"REF-EXT",
                        "ref_index": len(citations) + 1,
                        "doc_title": doc_name,
                        "file_name": "",
                        "page_number": page_no,
                        "section": sec_name,
                        "hierarchy_path": f"{doc_name} > {sec_name}",
                        "excerpt": "",
                        "rerank_score": 0.5
                    })

        # If LLM didn't produce bracketed tags, only attach top chunk references
        # if the answer does NOT indicate that the information is absent/not specified
        negative_indicators = [
            "not specified",
            "not mentioned",
            "no information",
            "cannot be answered",
            "not provided",
            "unavailable",
            "does not specify"
        ]
        is_negative = any(ind in answer_text.lower() for ind in negative_indicators)

        if not citations and retrieved_chunks and not is_negative:
            for idx, c in enumerate(retrieved_chunks[:2], 1):
                meta = c.get("metadata", {})
                citations.append({
                    "ref_id": f"REF-{idx}",
                    "ref_index": idx,
                    "doc_title": meta.get("doc_title", "Document"),
                    "file_name": meta.get("file_name", ""),
                    "page_number": meta.get("page_number", 1),
                    "section": meta.get("section", "General"),
                    "hierarchy_path": meta.get("hierarchy_path", ""),
                    "excerpt": c.get("text", "")[:350],
                    "rerank_score": round(c.get("rerank_score", 0.0), 3)
                })

        return answer_text, citations
