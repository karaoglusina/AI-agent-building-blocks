# Module 07: Text Preparation

> *"Every project starts with cleaning and chunking text"*

This module covers essential text preprocessing techniques that form the foundation of any NLP or AI pipeline.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_text_cleaning.py` | Basic Text Cleaning | Clean text before processing - garbage in, garbage out |
| `02_tokenization_nltk.py` | Tokenization with NLTK | Tokenization is the first step in most NLP pipelines |
| `03_lemmatization_stemming.py` | Lemmatization & Stemming | Lemmatization is accurate; stemming is faster but cruder |
| `04_stopwords.py` | Stopword Removal | Reduces noise but can lose context - use judiciously |
| `05_sentence_segmentation.py` | Sentence Segmentation | Sentence boundaries are tricky - "Dr. Smith" shouldn't split |
| `06_chunking_fixed.py` | Fixed-Size Chunking | Overlap prevents losing context at chunk boundaries |
| `07_chunking_semantic.py` | Semantic Chunking | Respect document structure - don't split mid-paragraph |
| `08_chunking_recursive.py` | Recursive Chunking | Try multiple separators until chunks fit size limits |
| `09_metadata_extraction.py` | Metadata Extraction | Metadata enables filtering and improves retrieval |

## Prerequisites

Install the required libraries:

```bash
pip install nltk spacy tiktoken
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_text_cleaning.py
python 02_tokenization_nltk.py
# ... etc
```

## Book References

- `AI_eng.8` - Data Engineering (cleaning, preprocessing)
- `AI_eng.6` - RAG (chunking strategies)
- `NLP_cook.1` - Text preprocessing fundamentals
- `speach_lang.I.2` - Text normalization
