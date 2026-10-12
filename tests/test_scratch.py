import sys
sys.path.insert(0, '.')
import re
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.semantic_chunker import SemanticChunker

# Monkey-patch or test
SemanticChunker.SECTION_SPLIT_REGEX = re.compile(r'\n+(?=(?:\d+\.|\d+\.\d+|Section\s+\d+|#{1,4})\s+[A-Z])')

p = PDFParser('sample_docs/TechCorp_Employee_Handbook_2026.pdf')
parsed = p.parse()
chunker = SemanticChunker()
chunks = chunker.chunk_document(parsed)

print(f"Generated {len(chunks)} chunks:")
for idx, c in enumerate(chunks):
    print(f"Chunk {idx}:")
    print(f"  Section: {c['section']}")
    print(f"  Tokens: {c['token_count']}")
    print(f"  Text preview: {repr(c['text'][:100])}")
    print()
