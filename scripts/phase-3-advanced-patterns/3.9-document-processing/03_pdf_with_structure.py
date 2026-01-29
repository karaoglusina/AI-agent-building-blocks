"""
03 - Structured PDF Extraction
===============================
Extract PDFs while preserving document structure (headings, paragraphs, lists).

Key concept: Structure-aware extraction enables better chunking and understanding - preserve document hierarchy.

Book reference: AI_eng.8
"""

import fitz
import re
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def extract_with_structure(pdf_path: str) -> dict:
    """Extract PDF with structure detection."""
    doc = fitz.open(pdf_path)

    structured_content = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]])

                    # Detect element type based on formatting
                    font_size = line["spans"][0]["size"] if line["spans"] else 12
                    font_flags = line["spans"][0]["flags"] if line["spans"] else 0

                    is_bold = font_flags & 2 ** 4  # Bold flag
                    is_large = font_size > 14

                    # Classify element
                    element_type = "paragraph"
                    if is_large and is_bold:
                        element_type = "heading"
                    elif text.strip().startswith(("•", "-", "·", "*")):
                        element_type = "list_item"
                    elif re.match(r"^\d+\.", text.strip()):
                        element_type = "numbered_list"

                    structured_content.append({
                        "type": element_type,
                        "text": text.strip(),
                        "page": page_num + 1,
                        "font_size": font_size
                    })

    doc.close()

    return {
        "elements": structured_content,
        "total_elements": len(structured_content)
    }


def group_by_sections(structured_content: list[dict]) -> list[dict]:
    """Group content into sections based on headings."""
    sections = []
    current_section = None

    for element in structured_content:
        if element["type"] == "heading":
            # Start new section
            if current_section:
                sections.append(current_section)

            current_section = {
                "title": element["text"],
                "page": element["page"],
                "content": []
            }
        elif current_section:
            # Add to current section
            current_section["content"].append(element)
        else:
            # Content before first heading
            if not sections or sections[0].get("title") != "Preamble":
                sections.insert(0, {
                    "title": "Preamble",
                    "page": element["page"],
                    "content": []
                })
            sections[0]["content"].append(element)

    # Add last section
    if current_section:
        sections.append(current_section)

    return sections


def extract_tables(pdf_path: str, page_num: int = 0) -> list[list[str]]:
    """Attempt to extract table-like structures."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Get text with position info
    blocks = page.get_text("blocks")

    # Simple table detection (requires actual table extraction library for production)
    # This is a simplified example

    tables = []
    current_row = []
    last_y = None

    for block in blocks:
        if len(block) >= 5:
            text = block[4].strip()
            y = block[1]  # y-coordinate

            # New row if y changes significantly
            if last_y and abs(y - last_y) > 10:
                if current_row:
                    tables.append(current_row)
                current_row = [text]
            else:
                current_row.append(text)

            last_y = y

    if current_row:
        tables.append(current_row)

    doc.close()
    return tables


if __name__ == "__main__":
    print("=== STRUCTURED PDF EXTRACTION ===\n")

    print("Structure-aware extraction preserves:")
    print("- Headings (large, bold text)")
    print("- Paragraphs (body text)")
    print("- Lists (bulleted, numbered)")
    print("- Sections (grouped by headings)")
    print("- Tables (layout-based detection)\n")

    # Example usage (requires actual PDF)
    # result = extract_with_structure("document.pdf")
    # sections = group_by_sections(result["elements"])
    #
    # for section in sections:
    #     print(f"\n=== {section['title']} (Page {section['page']}) ===")
    #     for element in section["content"][:3]:
    #         print(f"{element['type']}: {element['text'][:80]}...")

    print("To use:")
    print('  result = extract_with_structure("your_file.pdf")')
    print('  sections = group_by_sections(result["elements"])')
    print('  # Now you have structured sections!')

    print("\nKey insight: Structure preservation = better understanding and chunking")
