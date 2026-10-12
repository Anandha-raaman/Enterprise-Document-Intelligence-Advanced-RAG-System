import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader

class PDFParser:
    """
    Robust enterprise PDF parser extracting text, structural headers, 
    page numbers, and hierarchical document context.
    """

    # Heading detection patterns
    HEADING_PATTERNS = [
        re.compile(r"^(?:(?:Section|\xA7)\s*)?(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,80})$", re.MULTILINE),
        re.compile(r"^(\d+\.\s+[A-Z][^\n]{3,80})$", re.MULTILINE),
        re.compile(r"^(?:#{1,4}\s+)?([A-Z][A-Z0-9\s,\-\:]{4,70})$", re.MULTILINE),
        re.compile(r"^([A-Z][a-zA-Z0-9\s,\-\:]{3,60}:)$", re.MULTILINE),
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        self.doc_hash = self._compute_hash()
        self.doc_title = self._infer_doc_title()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash for document version tracking."""
        hasher = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _infer_doc_title(self) -> str:
        """Derive document title from filename or metadata."""
        stem = self.file_path.stem
        # Clean underscores and hyphens into clean title
        title = re.sub(r"[_\-]+", " ", stem).strip()
        return title.title()

    def parse(self) -> Dict[str, Any]:
        """
        Parse PDF page-by-page, identifying structural sections and hierarchy.
        Returns document metadata and list of structured pages.
        """
        reader = PdfReader(str(self.file_path))
        num_pages = len(reader.pages)
        
        pages_data: List[Dict[str, Any]] = []
        current_section = "Introduction"
        current_subsection = ""

        # Extract PDF metadata if present
        pdf_meta = reader.metadata or {}
        extracted_title = pdf_meta.get("/Title")
        if extracted_title and len(str(extracted_title).strip()) > 3 and not str(extracted_title).strip().lower().startswith("(anonymous"):
            self.doc_title = str(extracted_title).strip()

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            
            # Clean non-printable / control characters while preserving formatting
            cleaned_text = self._clean_text(raw_text)
            
            # Extract section markers on this page
            page_sections = self._extract_sections(cleaned_text)
            if page_sections:
                # Update current section hierarchy
                current_section = page_sections[0].get("section", current_section)
                if len(page_sections) > 1:
                    current_subsection = page_sections[-1].get("section", "")
                else:
                    current_subsection = page_sections[0].get("subsection", "")

            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text,
                "detected_sections": page_sections,
                "active_section": current_section,
                "active_subsection": current_subsection
            })

        return {
            "file_name": self.file_path.name,
            "doc_id": f"{self.file_path.stem.lower()}_{self.doc_hash}",
            "doc_hash": self.doc_hash,
            "doc_title": self.doc_title,
            "total_pages": num_pages,
            "pages": pages_data
        }

    def _clean_text(self, text: str) -> str:
        """Normalize line breaks, hyphenated line wraps, and whitespace."""
        # Fix hyphenated words at line endings: e.g. "commer-\ncial" -> "commercial"
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize multiple spaces (preserving newlines)
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize consecutive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Identify headers or section labels within page text."""
        sections = []
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) > 85:
                continue

            for pattern in self.HEADING_PATTERNS:
                match = pattern.match(line_str)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        sec_num, sec_title = groups
                        sections.append({
                            "section": f"{sec_num} {sec_title}".strip(),
                            "subsection": ""
                        })
                    elif len(groups) == 1:
                        sections.append({
                            "section": groups[0].strip(),
                            "subsection": ""
                        })
                    break
        return sections
