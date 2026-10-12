import re
from typing import List, Dict, Any
from app.config import Config

class SemanticChunker:
    """
    Semantic chunker designed for enterprise documents.
    Enforces target token length (~500 tokens), 10% overlap (~50 tokens),
    and boundary preservation at paragraph and sentence levels.
    """

    def __init__(
        self,
        target_tokens: int = Config.TARGET_CHUNK_TOKENS,
        overlap_tokens: int = Config.CHUNK_OVERLAP_TOKENS,
        min_tokens: int = Config.MIN_CHUNK_TOKENS
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        # Regex for sentence splitting
        self.sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count based on whitespace & punctuation heuristics.
        Approximates ~1.3 tokens per word or ~4 chars per token.
        """
        words = text.split()
        if not words:
            return 0
        return max(1, int(len(words) * 1.3))

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes output of PDFParser, executes semantic chunking across pages,
        and constructs hierarchical metadata for each chunk.
        """
        chunks: List[Dict[str, Any]] = []
        doc_id = parsed_doc["doc_id"]
        doc_title = parsed_doc["doc_title"]
        file_name = parsed_doc["file_name"]
        doc_hash = parsed_doc["doc_hash"]

        # Flatten document into semantic units (sentences/paragraphs) tagged with page and section
        units: List[Dict[str, Any]] = []
        section_split_regex = re.compile(r"\n+(?=(?:\d+\.|\d+\.\d+|Section\s+\d+|#{1,4})\s+[A-Z])")

        for page_data in parsed_doc["pages"]:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            active_section = page_data.get("active_section") or "General"
            active_subsection = page_data.get("active_subsection") or ""

            if not page_text.strip():
                continue

            # Split page text into distinct section blocks
            raw_blocks = section_split_regex.split(page_text.strip())

            for block in raw_blocks:
                b_clean = block.strip()
                if not b_clean:
                    continue

                lines = b_clean.split("\n")
                first_line = lines[0].strip()

                # Check if block starts with a section header
                if re.match(r"^(?:\d+\.|\d+\.\d+|Section\s+\d+|#{1,4})\s+[A-Z]", first_line):
                    active_section = first_line
                
                content = b_clean

                # Split block into sentences
                sentences = self.sentence_regex.split(content)
                for sentence in sentences:
                    s_clean = sentence.strip()
                    if s_clean:
                        units.append({
                            "text": s_clean,
                            "page": page_num,
                            "section": active_section,
                            "subsection": active_subsection,
                            "tokens": self.estimate_tokens(s_clean)
                        })

        if not units:
            return []

        # Sliding window with semantic boundary assembly & section isolation
        i = 0
        chunk_idx = 0
        total_units = len(units)

        while i < total_units:
            current_chunk_units: List[Dict[str, Any]] = []
            current_tokens = 0
            start_unit_idx = i
            stopped_at_section_boundary = False

            while i < total_units:
                unit = units[i]
                unit_tokens = unit["tokens"]

                # Isolate chunks by major section: break whenever a new section begins
                if current_chunk_units and unit["section"] != current_chunk_units[0]["section"]:
                    stopped_at_section_boundary = True
                    break

                # If adding this unit exceeds target tokens and we already have sufficient tokens
                if current_tokens + unit_tokens > self.target_tokens and current_tokens >= self.min_tokens:
                    break

                current_chunk_units.append(unit)
                current_tokens += unit_tokens
                i += 1

            if not current_chunk_units:
                # Force include at least one unit even if it exceeds target
                current_chunk_units.append(units[i])
                current_tokens += units[i]["tokens"]
                i += 1

            # Assemble chunk text
            chunk_text = " ".join([u["text"] for u in current_chunk_units])
            start_page = current_chunk_units[0]["page"]
            end_page = current_chunk_units[-1]["page"]
            primary_section = current_chunk_units[0]["section"]
            primary_subsection = current_chunk_units[0]["subsection"]

            # Construct hierarchical breadcrumb path
            path_parts = [doc_title]
            if primary_section:
                path_parts.append(primary_section)
            if primary_subsection and primary_subsection != primary_section:
                path_parts.append(primary_subsection)
            hierarchy_path = " > ".join(path_parts)

            chunk_id = f"{doc_id}_c{chunk_idx:04d}"

            chunk_data = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "file_name": file_name,
                "doc_hash": doc_hash,
                "text": chunk_text,
                "page_number": start_page,
                "page_end": end_page,
                "section": primary_section or "General",
                "subsection": primary_subsection or "",
                "hierarchy_path": hierarchy_path,
                "chunk_index": chunk_idx,
                "token_count": current_tokens,
                "char_count": len(chunk_text)
            }
            chunks.append(chunk_data)
            chunk_idx += 1

            # Determine rollback index for overlap (~10% tokens), strictly within same section
            if i < total_units and not stopped_at_section_boundary:
                overlap_accum = 0
                rollback_count = 0
                target_sec = current_chunk_units[-1]["section"]
                for u in reversed(current_chunk_units):
                    if u["section"] != target_sec:
                        break
                    overlap_accum += u["tokens"]
                    rollback_count += 1
                    if overlap_accum >= self.overlap_tokens:
                        break
                
                # Rollback i by rollback_count (ensuring progress)
                new_i = max(start_unit_idx + 1, i - rollback_count)
                i = new_i

        return chunks
