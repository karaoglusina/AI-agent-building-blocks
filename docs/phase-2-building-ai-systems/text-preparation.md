# Text Preparation

Every project starts here. Before embeddings, before RAG, before any of the interesting stuff - you need clean, well-structured text. Garbage in, garbage out.

## Why This Matters

Raw text is messy. Job postings have HTML tags, weird Unicode characters, inconsistent whitespace. Long documents need to be chunked for embedding. None of the downstream components work well with messy input.

For our job market analyzer, text preparation turns raw job descriptions into clean, chunked text ready for embedding and search.

## The Key Ideas

### Cleaning

Basic cleaning removes the obvious garbage:

```python
import re

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Fix encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text
```

Simple, but necessary. You'd be surprised how often HTML tags sneak into "plain text" fields.

### Tokenization

Breaking text into words or sentences. NLTK and spaCy both do this well:

```python
import nltk
tokens = nltk.word_tokenize("Machine learning is fascinating.")
# ['Machine', 'learning', 'is', 'fascinating', '.']
```

Tokenization matters for token counting, keyword extraction, and understanding text structure.

### Lemmatization vs Stemming

Both reduce words to base forms, but differently:
- **Stemming**: Fast, crude. "running" → "run", "better" → "better" (or "bett")
- **Lemmatization**: Slower, accurate. "running" → "run", "better" → "good"

Use lemmatization when accuracy matters (search, extraction). Use stemming when you need speed.

### Chunking

Long documents need to be split for embedding. The trick is doing it without losing context.

**Fixed-size chunking** is simplest:
```python
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

The overlap prevents losing context at boundaries.

**Semantic chunking** is smarter - split at paragraph or section boundaries:
```python
def semantic_chunk(text):
    # Split on double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    # Merge small paragraphs, split large ones
    # ...
```

**Recursive chunking** tries multiple separators until chunks fit:
```python
# Try to split on "\n\n" first
# If chunks too big, split on "\n"
# If still too big, split on ". "
# Last resort: split on " "
```

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_text_cleaning.py | Remove HTML, normalize whitespace, fix encoding |
| 02_tokenization_nltk.py | Word and sentence tokenization |
| 03_lemmatization_stemming.py | Reduce words to base forms |
| 04_stopwords.py | Remove common words |
| 05_sentence_segmentation.py | Split text into sentences properly |
| 06_chunking_fixed.py | Fixed-size chunks with overlap |
| 07_chunking_semantic.py | Chunk by paragraphs and sections |
| 08_chunking_recursive.py | Try multiple separators |
| 09_metadata_extraction.py | Extract titles, dates, structure |

## Things to Think About

- **When does cleaning remove useful information?** Be careful not to over-clean. "C++" is valid, even if it looks like noise.
- **How big should chunks be?** Depends on your use case. Smaller chunks = more precise retrieval but less context. Larger chunks = more context but less precise. 200-500 tokens is a common starting point.
- **Why not just embed whole documents?** Long text gets truncated by embedding models. Even if it fits, retrieval works better with focused chunks.

## Related

- [Embeddings](../phase-1-foundations/embeddings.md) - What happens after cleaning
- [RAG Pipeline](./rag-pipeline.md) - Where clean text gets used
- [Information Extraction](./information-extraction.md) - Getting structure from text

## Book References

- AI_eng.8 - Data engineering
- AI_eng.6 - Chunking for RAG
- NLP_cook.1 - Text preprocessing
- speach_lang.I.2 - Text normalization
