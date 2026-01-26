"""
01 - PDF with PyMuPDF
======================
Extract text from PDFs using PyMuPDF (fitz).

Key concept: PyMuPDF is fast and accurate for text extraction - handles complex layouts well.

Book reference: AI_eng.8
"""

import fitz  # PyMuPDF
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def extract_text_pymupdf(pdf_path: str) -> dict:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)

    pages = []
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })

        full_text.append(text)

    doc.close()

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text),
        "total_chars": sum(p["char_count"] for p in pages)
    }


def extract_with_metadata(pdf_path: str) -> dict:
    """Extract text plus document metadata."""
    doc = fitz.open(pdf_path)

    # Get metadata
    metadata = doc.metadata

    # Extract text
    text_data = extract_text_pymupdf(pdf_path)

    doc.close()

    return {
        "metadata": metadata,
        "text_data": text_data
    }


def extract_by_blocks(pdf_path: str, page_num: int = 0) -> list[dict]:
    """Extract text blocks (maintains layout structure)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Get text blocks
    blocks = page.get_text("blocks")

    block_list = []
    for block in blocks:
        # block format: (x0, y0, x1, y1, "text", block_no, block_type)
        if len(block) >= 5:
            block_list.append({
                "text": block[4].strip(),
                "bbox": (block[0], block[1], block[2], block[3]),
                "type": "text" if len(block) < 7 else block[6]
            })

    doc.close()
    return block_list


if __name__ == "__main__":
    # Note: This script requires a PDF file to work
    # Create a sample PDF or point to an existing one

    print("=== PDF EXTRACTION WITH PYMUPDF ===\n")
    print("PyMuPDF (fitz) features:")
    print("- Fast extraction")
    print("- Preserves layout")
    print("- Handles complex PDFs")
    print("- Extracts metadata")
    print("- Block-level extraction\n")

    # Example usage (requires actual PDF file)
    # result = extract_text_pymupdf("sample.pdf")
    # print(f"Extracted {result['total_pages']} pages")
    # print(f"Total characters: {result['total_chars']}")
    # print(f"\nFirst 500 chars:\n{result['full_text'][:500]}")

    print("To use:")
    print('  result = extract_text_pymupdf("your_file.pdf")')
    print('  print(result["full_text"])')

    print("\nKey insight: PyMuPDF is best for speed and accuracy")
