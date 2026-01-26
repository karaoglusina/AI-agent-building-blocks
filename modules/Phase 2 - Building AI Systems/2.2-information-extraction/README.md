# Module 08: Information Extraction

> *"Extract structure from unstructured text"*

This module covers techniques for extracting specific information from text, from rule-based methods to LLM-powered extraction.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_ner_spacy.py` | NER with spaCy | NER identifies real-world objects in text |
| `02_ner_custom.py` | Custom Entity Patterns | Standard NER misses domain terms - extend with patterns |
| `03_pos_tagging.py` | Part-of-Speech Tagging | POS tags reveal word roles - nouns are things, verbs are actions |
| `04_grammar_noun_chunks.py` | Noun Chunks & Grammar | Noun chunks capture meaningful phrases, not just words |
| `05_keywords_tfidf.py` | TF-IDF Keywords | TF-IDF finds words important in a document but rare overall |
| `06_keywords_keybert.py` | KeyBERT Extraction | KeyBERT uses embeddings to find semantically relevant keywords |
| `07_lexicon_approaches.py` | Lexicon & Dictionary Methods | Lexicons provide interpretable, domain-specific extraction |
| `08_extraction_llm.py` | LLM-Based Extraction | LLMs understand context and extract nuanced information |
| `09_regex_patterns.py` | Regex Pattern Matching | Regex is fast and precise for well-defined patterns |
| `10_fuzzy_matching.py` | Fuzzy String Matching | Fuzzy matching finds similar strings despite spelling differences |

## Job Data Application

All scripts use the job posting dataset to demonstrate extraction:
- Extract skills, companies, locations from job descriptions
- Identify required experience and education
- Normalize and deduplicate skill mentions

## Prerequisites

Install the required libraries:

```bash
pip install spacy nltk keybert scikit-learn textblob rapidfuzz openai pydantic
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('opinion_lexicon')"
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_ner_spacy.py
python 02_ner_custom.py
# ... etc
```

## Extraction Method Comparison

| Method | Speed | Accuracy | Flexibility | Use Case |
|--------|-------|----------|-------------|----------|
| Regex | Fast | High (for patterns) | Low | Emails, phones, URLs |
| spaCy NER | Fast | Good | Medium | Standard entities |
| Custom NER | Fast | Good | High | Domain entities |
| TF-IDF | Fast | Medium | Low | Keyword discovery |
| KeyBERT | Medium | Good | Medium | Semantic keywords |
| Lexicon | Fast | Medium | High | Sentiment, categories |
| LLM | Slow | High | Very High | Complex extraction |
| Fuzzy Match | Medium | Good | Medium | Normalization |

## Book References

- `NLP_cook.5` - Information extraction
- `speach_lang.III.17` - Named entity recognition
- `speach_lang.I.2` - Regular expressions
- `AI_eng.5` - LLM-based extraction
- `hands_on_LLM.II.6` - Structured output
