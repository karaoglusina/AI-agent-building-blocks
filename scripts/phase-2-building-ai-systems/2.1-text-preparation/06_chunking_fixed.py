"""
06 - Fixed-Size Chunking
========================
Split text by character or token count with overlap.

Key concept: Overlap prevents losing context at chunk boundaries.

Book reference: AI_eng.6 (RAG)
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import tiktoken
except ImportError:
    MISSING_DEPENDENCIES.append('tiktoken')

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs

# Skip if dependencies missing in TEST_MODE
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)


def chunk_by_chars(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed character-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move back by overlap amount
    return chunks


def chunk_by_tokens(text: str, chunk_size: int = 200, overlap: int = 20, 
                    model: str = "gpt-4o") -> list[str]:
    """Split text into fixed token-size chunks with overlap."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap
    return chunks


if __name__ == "__main__":
    # Load a job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"]
    
    print(f"=== ORIGINAL TEXT ===")
    print(f"Length: {len(text)} characters")
    print()
    
    # Character-based chunking
    char_chunks = chunk_by_chars(text, chunk_size=400, overlap=50)
    print(f"=== CHARACTER CHUNKS (400 chars, 50 overlap) ===")
    print(f"Number of chunks: {len(char_chunks)}")
    for i, chunk in enumerate(char_chunks[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  Start: {chunk[:60]}...")
        print(f"  End: ...{chunk[-60:]}")
    print()
    
    # Token-based chunking
    token_chunks = chunk_by_tokens(text, chunk_size=100, overlap=10)
    print(f"=== TOKEN CHUNKS (100 tokens, 10 overlap) ===")
    print(f"Number of chunks: {len(token_chunks)}")
    
    enc = tiktoken.encoding_for_model("gpt-4o")
    for i, chunk in enumerate(token_chunks[:3], 1):
        token_count = len(enc.encode(chunk))
        print(f"\nChunk {i} ({token_count} tokens):")
        print(f"  {chunk[:100]}...")
