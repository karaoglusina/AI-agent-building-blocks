"""
04 - PDF to Searchable Chunks
==============================
Full pipeline: PDF → structured extraction → chunking → indexing.

Key concept: End-to-end pipeline from PDF to searchable knowledge base - production-ready document processing.

Book reference: AI_eng.6, AI_eng.8
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import fitz
try:
    import chromadb
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

from typing import List
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from PDF with page info."""
    doc = fitz.open(pdf_path)

    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages_text.append({
                "page_number": page_num + 1,
                "text": text
            })

    doc.close()
    return pages_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Chunk text with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


def create_chunks_from_pdf(pdf_path: str, chunk_size: int = 500) -> list[dict]:
    """Create chunks from PDF with metadata."""
    pages = extract_pdf_text(pdf_path)

    all_chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size=chunk_size)

        for i, chunk_text in enumerate(page_chunks):
            if chunk_text.strip():
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page": page["page_number"],
                    "chunk_on_page": i + 1,
                    "source": pdf_path
                })
                chunk_id += 1

    return all_chunks


def index_chunks(chunks: list[dict], collection_name: str = "pdf_docs") -> chromadb.Collection:
    """Index chunks in ChromaDB."""
    client = chromadb.Client()

    # Create or get collection
    try:
        collection = client.get_collection(collection_name)
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(collection_name)

    # Prepare data
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "page": chunk["page"],
            "chunk_on_page": chunk["chunk_on_page"],
            "source": chunk["source"]
        }
        for chunk in chunks
    ]
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]

    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection


def search_pdf_content(collection: chromadb.Collection, query: str, n_results: int = 3) -> list[dict]:
    """Search indexed PDF content."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    search_results = []
    for i in range(len(results['ids'][0])):
        search_results.append({
            "text": results['documents'][0][i],
            "page": results['metadatas'][0][i]["page"],
            "source": results['metadatas'][0][i]["source"],
            "similarity": 1 - results['distances'][0][i]
        })

    return search_results


def full_pipeline_demo():
    """Demonstrate full PDF processing pipeline."""
    print("=== PDF TO SEARCHABLE CHUNKS PIPELINE ===\n")

    print("Pipeline steps:")
    print("1. Extract text from PDF (with page numbers)")
    print("2. Chunk text (with overlap for context)")
    print("3. Add metadata (page, source, chunk ID)")
    print("4. Generate embeddings")
    print("5. Index in vector database")
    print("6. Enable semantic search\n")

    print("Example usage:\n")
    print('# Process PDF')
    print('chunks = create_chunks_from_pdf("document.pdf", chunk_size=500)')
    print(f'print(f"Created {{len(chunks)}} chunks")\n')

    print('# Index chunks')
    print('collection = index_chunks(chunks, "my_docs")')
    print(f'print(f"Indexed {{collection.count()}} chunks")\n')

    print('# Search')
    print('results = search_pdf_content(collection, "machine learning")')
    print('for r in results:')
    print('    print(f"Page {r[\'page\']}: {r[\'text\'][:100]}...")\n')

    print("Production considerations:")
    print("- OCR for scanned PDFs (use pytesseract)")
    print("- Table extraction (use camelot or tabula)")
    print("- Image extraction (use fitz.extract_images)")
    print("- Metadata enrichment (dates, authors, titles)")
    print("- Deduplication across documents")
    print("- Batch processing for multiple PDFs")


if __name__ == "__main__":
    full_pipeline_demo()

    print("\n" + "=" * 70)
    print("\nKey insight: Full pipeline = PDF → searchable knowledge base")
    print("This pattern works for:")
    print("- Technical documentation")
    print("- Research papers")
    print("- Reports and contracts")
    print("- Books and manuals")
    print("- Any long-form documents")
