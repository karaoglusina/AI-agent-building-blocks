# Module 3.9: Document Processing

> *"Handle PDFs and complex documents"*

This module covers extracting text from PDFs, preserving structure, and building end-to-end pipelines from documents to searchable knowledge bases.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_pdf_pymupdf.py` | PyMuPDF Extraction | Fast, accurate PDF text extraction with PyMuPDF (fitz) |
| `02_pdf_pypdf.py` | pypdf Extraction | Pure Python PDF extraction - easy installation |
| `03_pdf_with_structure.py` | Structured Extraction | Preserve headings, paragraphs, lists - maintain hierarchy |
| `04_pdf_to_chunks.py` | PDF to Searchable Chunks | Full pipeline from PDF to indexed knowledge base |

## Why Document Processing?

Most enterprise knowledge is locked in PDFs:
- Technical documentation
- Research papers
- Contracts and legal docs
- Reports and presentations
- Books and manuals

Extracting and indexing this content enables:
- Semantic search
- Q&A systems
- Document summarization
- Information extraction

## Library Comparison

### PyMuPDF (fitz)
- **Speed**: ⚡⚡⚡ Very fast
- **Accuracy**: ✓✓✓ Excellent
- **Complex PDFs**: ✓✓✓ Handles well
- **Installation**: Requires C bindings
- **Best for**: Production systems, complex docs

### pypdf
- **Speed**: ⚡ Moderate
- **Accuracy**: ✓✓ Good for simple PDFs
- **Complex PDFs**: ✓ Basic support
- **Installation**: Pure Python, easy
- **Best for**: Simple PDFs, prototyping

## Extraction Strategies

### 1. Simple Text Extraction
```python
doc = fitz.open("file.pdf")
text = ""
for page in doc:
    text += page.get_text()
```

**Use when**: Simple docs, just need text

### 2. Page-by-Page
```python
pages = []
for i, page in enumerate(doc):
    pages.append({
        "number": i + 1,
        "text": page.get_text()
    })
```

**Use when**: Need page context for citations

### 3. Structure-Aware
```python
blocks = page.get_text("dict")
# Analyze font size, style to detect headings
```

**Use when**: Need to preserve document structure

### 4. Block-Level
```python
blocks = page.get_text("blocks")
# Each block is a text region
```

**Use when**: Complex layouts, columns

## Production Pipeline

```
PDF → Extract (PyMuPDF)
    → Detect structure (headings, sections)
    → Chunk (500-1000 chars with overlap)
    → Add metadata (page, source, section)
    → Generate embeddings
    → Index (ChromaDB/PostgreSQL)
    → Enable search
```

## Chunking Strategies

### Fixed-Size with Overlap
```python
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunks.append(text[i:i + chunk_size])
```

**Pros**: Simple, consistent size
**Cons**: May split mid-sentence

### Sentence-Aware
```python
sentences = sent_tokenize(text)
# Group sentences to reach chunk_size
```

**Pros**: Respects boundaries
**Cons**: Variable size

### Section-Based
```python
# Chunk by headings/sections
for section in sections:
    chunks.append(section["content"])
```

**Pros**: Semantically meaningful
**Cons**: Very variable size

## Metadata Enrichment

### Essential Metadata
- **Page number**: For citations
- **Source file**: Original document
- **Section/heading**: Context
- **Chunk ID**: Unique identifier

### Optional Metadata
- **Author**: Document creator
- **Date**: Creation/modification date
- **Document type**: Report, paper, manual
- **Keywords**: Extracted topics

## Prerequisites

Install required libraries:

```bash
pip install pymupdf pypdf chromadb
```

## Running the Scripts

```bash
python 01_pdf_pymupdf.py
python 02_pdf_pypdf.py
python 03_pdf_with_structure.py
python 04_pdf_to_chunks.py
```

**Note**: These scripts need actual PDF files to process. Use your own PDFs or create test files.

## Common Challenges

### 1. Scanned PDFs (Images)
**Solution**: Use OCR (pytesseract + Tesseract)

```python
import pytesseract
from PIL import Image

# Extract images from PDF
# Run OCR on each image
text = pytesseract.image_to_string(image)
```

### 2. Tables
**Solution**: Use specialized libraries

```python
# camelot-py for table extraction
import camelot
tables = camelot.read_pdf("file.pdf")
```

### 3. Multi-Column Layouts
**Solution**: Use block-level extraction + sort by position

```python
blocks = page.get_text("blocks")
sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
```

### 4. Mathematical Formulas
**Solution**: LaTeX-based extraction or image extraction

```python
# Extract images, use vision models to parse
images = page.get_images()
```

## Advanced Techniques

### Table Extraction
```bash
pip install camelot-py[cv]  # OpenCV method
pip install tabula-py        # Alternative
```

### Image Extraction
```python
images = page.get_images()
for img in images:
    xref = img[0]
    base_image = doc.extract_image(xref)
    image_data = base_image["image"]
```

### OCR for Scanned PDFs
```bash
pip install pytesseract
# Install Tesseract: brew install tesseract (Mac)
```

### Layout Analysis
```bash
pip install layout-parser
# Detect document structure with ML
```

## Best Practices

1. **Test extraction quality** - Manually check samples
2. **Handle errors gracefully** - Some PDFs are corrupt/protected
3. **Preserve metadata** - Page numbers enable citations
4. **Chunk wisely** - Balance size vs context
5. **Deduplicate** - Same content may appear in multiple docs
6. **Version documents** - Track updates to source files

## Performance Tips

### For Large PDFs
- Process pages in parallel
- Stream large files
- Use incremental indexing

### For Many PDFs
- Batch processing
- Distributed processing (Celery)
- Cache extracted text

### For Fast Search
- Optimize chunk size (not too small, not too large)
- Use hybrid search (keyword + semantic)
- Filter by metadata first

## Book References

- `AI_eng.6` - RAG and document chunking strategies
- `AI_eng.8` - Data processing and quality

## Next Steps

After mastering document processing:
- Module 4.1: Docker & Containerization
- Module 4.2: PostgreSQL + pgvector (for large-scale indexing)
- Module 4.5: Async & Background Jobs (for batch processing)
