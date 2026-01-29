"""
02 - PDF with pypdf
===================
Extract text from PDFs using pypdf (pure Python).

Key concept: pypdf is pure Python, easier to install, good for simple PDFs - but slower than PyMuPDF for complex docs.

Book reference: AI_eng.8
"""

from pypdf import PdfReader
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def extract_text_pypdf(pdf_path: str) -> dict:
    """Extract text from PDF using pypdf."""
    reader = PdfReader(pdf_path)

    pages = []
    full_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })

        full_text.append(text)

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text),
        "total_chars": sum(p["char_count"] for p in pages)
    }


def extract_metadata_pypdf(pdf_path: str) -> dict:
    """Extract PDF metadata using pypdf."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata

    return {
        "title": metadata.get("/Title", ""),
        "author": metadata.get("/Author", ""),
        "subject": metadata.get("/Subject", ""),
        "creator": metadata.get("/Creator", ""),
        "producer": metadata.get("/Producer", ""),
        "creation_date": str(metadata.get("/CreationDate", "")),
        "pages": len(reader.pages)
    }


def extract_page_range(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extract specific page range."""
    reader = PdfReader(pdf_path)

    text_parts = []
    for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
        page = reader.pages[page_num]
        text_parts.append(page.extract_text())

    return "\n\n".join(text_parts)


def compare_libraries():
    """Compare PyMuPDF vs pypdf."""
    print("\n=== PYMUPDF VS PYPDF ===\n")

    comparison = {
        "PyMuPDF (fitz)": {
            "Speed": "⚡⚡⚡ Fast",
            "Accuracy": "✓✓✓ Excellent",
            "Complex PDFs": "✓✓✓ Handles well",
            "Installation": "Requires C bindings",
            "Best for": "Production, complex docs"
        },
        "pypdf": {
            "Speed": "⚡ Slower",
            "Accuracy": "✓✓ Good",
            "Complex PDFs": "✓ Basic support",
            "Installation": "Pure Python, easy",
            "Best for": "Simple PDFs, easy setup"
        }
    }

    for lib, features in comparison.items():
        print(f"{lib}:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")
        print()


if __name__ == "__main__":
    print("=== PDF EXTRACTION WITH PYPDF ===\n")
    print("pypdf features:")
    print("- Pure Python (easy install)")
    print("- Good for simple PDFs")
    print("- Metadata extraction")
    print("- Page range extraction\n")

    # Example usage (requires actual PDF file)
    # result = extract_text_pypdf("sample.pdf")
    # print(f"Extracted {result['total_pages']} pages")
    # print(f"Total characters: {result['total_chars']}")

    print("To use:")
    print('  result = extract_text_pypdf("your_file.pdf")')
    print('  print(result["full_text"])')

    # Compare libraries
    compare_libraries()

    print("Key insight: pypdf for simplicity, PyMuPDF for performance")
