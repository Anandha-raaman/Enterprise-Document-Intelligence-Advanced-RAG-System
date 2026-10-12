import re
from typing import Dict, Any, List

class MetadataExtractor:
    """
    Enriches chunks and document records with structural and domain metadata.
    """

    @staticmethod
    def extract_document_summary(parsed_doc: Dict[str, Any], max_words: int = 60) -> str:
        """Extract opening abstract or introductory passage as document overview."""
        pages = parsed_doc.get("pages", [])
        if not pages:
            return ""
        
        first_page_text = pages[0].get("text", "")
        words = first_page_text.split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + "..."

    @staticmethod
    def extract_domain_tags(text: str) -> List[str]:
        """Identify key enterprise domain topics (HR, Security, Compliance, Finance, SLA)."""
        tags = set()
        text_lower = text.lower()

        keywords = {
            "Policy & HR": ["leave", "absence", "handbook", "vacation", "wellness", "reimbursement", "employee"],
            "Information Security": ["mfa", "password", "authentication", "encryption", "soc2", "vulnerability", "firewall", "security"],
            "Cloud & IT Operations": ["sla", "uptime", "latency", "kubernetes", "api", "availability", "failover", "database"],
            "Compliance & Legal": ["compliance", "audit", "governance", "gdpr", "hipaa", "regulatory", "liability"],
            "Finance & Billing": ["per diem", "expense", "budget", "invoice", "receipt", "travel"]
        }

        for category, terms in keywords.items():
            if any(term in text_lower for term in terms):
                tags.add(category)

        return sorted(list(tags)) or ["General Documentation"]
