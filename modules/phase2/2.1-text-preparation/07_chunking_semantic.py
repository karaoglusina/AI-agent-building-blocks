"""
07 - Semantic Chunking
======================
Split by paragraphs, sections, or natural meaning boundaries.

Key concept: Respect document structure - don't split mid-sentence or mid-paragraph.

Book reference: AI_eng.6
"""

import re

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs


def chunk_by_paragraphs(text: str, min_length: int = 100) -> list[str]:
    """Split by paragraph breaks, merge short paragraphs."""
    # Split on double newlines or multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Merge short paragraphs
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < min_length:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    
    return chunks


def chunk_by_sections(text: str) -> list[dict]:
    """Split by section headers (markdown-style or common patterns)."""
    # Common section patterns in job descriptions
    section_patterns = [
        r'^#+\s+(.+)$',              # Markdown headers
        r'^([A-Z][A-Z\s]+):?\s*$',   # ALL CAPS HEADERS
        r'^(Requirements|Responsibilities|Qualifications|About|Benefits):?\s*',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in section_patterns)
    
    chunks = []
    current_section = "Introduction"
    current_content = []
    
    for line in text.split('\n'):
        is_header = False
        for pattern in section_patterns:
            if re.match(pattern, line.strip(), re.MULTILINE):
                # Save previous section
                if current_content:
                    chunks.append({
                        "section": current_section,
                        "content": '\n'.join(current_content).strip()
                    })
                current_section = line.strip()
                current_content = []
                is_header = True
                break
        
        if not is_header:
            current_content.append(line)
    
    # Don't forget last section
    if current_content:
        chunks.append({
            "section": current_section,
            "content": '\n'.join(current_content).strip()
        })
    
    return chunks


if __name__ == "__main__":
    # Load job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"]
    
    print("=== PARAGRAPH CHUNKING ===")
    para_chunks = chunk_by_paragraphs(text, min_length=200)
    print(f"Found {len(para_chunks)} paragraph chunks")
    for i, chunk in enumerate(para_chunks[:3], 1):
        print(f"\nParagraph {i} ({len(chunk)} chars):")
        print(f"  {chunk[:150]}...")
    print()
    
    print("=== SECTION CHUNKING ===")
    section_chunks = chunk_by_sections(text)
    print(f"Found {len(section_chunks)} sections")
    for chunk in section_chunks:
        content_preview = chunk['content'][:100].replace('\n', ' ')
        print(f"\n[{chunk['section']}]")
        print(f"  {content_preview}...")
