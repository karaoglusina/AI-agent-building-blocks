"""
08 - Recursive Chunking
=======================
LangChain-style recursive split - implemented from scratch.

Key concept: Try multiple separators in order until chunks fit size limits.

Book reference: AI_eng.6
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] = None
) -> list[str]:
    """
    Recursively split text using multiple separators.
    
    Tries each separator in order:
    1. Double newline (paragraphs)
    2. Single newline (lines)
    3. Period + space (sentences)
    4. Space (words)
    5. Empty string (characters)
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    chunks = []
    
    def split_text(text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text with current separator."""
        if sep_index >= len(separators):
            # Base case: can't split further, return as-is
            return [text]
        
        separator = separators[sep_index]
        
        if not separator:  # Empty separator = split by character
            parts = list(text)
        else:
            parts = text.split(separator)
        
        result = []
        current = ""
        
        for part in parts:
            # Add separator back (except for last part)
            part_with_sep = part + separator if separator else part
            
            if len(current) + len(part_with_sep) <= chunk_size:
                current += part_with_sep
            else:
                if current:
                    result.append(current.strip())
                
                # If this single part is too large, recurse with next separator
                if len(part_with_sep) > chunk_size:
                    result.extend(split_text(part, sep_index + 1))
                    current = ""
                else:
                    current = part_with_sep
        
        if current.strip():
            result.append(current.strip())
        
        return result
    
    raw_chunks = split_text(text)
    
    # Add overlap between chunks
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            # Get overlap from previous chunk
            prev_end = raw_chunks[i-1][-chunk_overlap:]
            chunk = prev_end + " " + chunk
        final_chunks.append(chunk)
    
    return final_chunks


if __name__ == "__main__":
    # Load job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"]
    
    print(f"=== ORIGINAL TEXT ===")
    print(f"Length: {len(text)} characters")
    print()
    
    # Chunk with different sizes
    for size in [300, 500]:
        chunks = recursive_chunk(text, chunk_size=size, chunk_overlap=30)
        print(f"=== CHUNKS (size={size}, overlap=30) ===")
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i} ({len(chunk)} chars):")
            # Show that chunks don't split mid-sentence
            if ". " in chunk:
                print(f"  Ends with: ...{chunk[-80:]}")
        print()
