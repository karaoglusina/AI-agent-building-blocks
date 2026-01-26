# Information Extraction

Text contains information. Job postings mention skills, locations, salaries. News articles name people and companies. The challenge is pulling this structure out of unstructured text.

## Why This Matters

Raw text isn't very useful in a database. You can't filter job postings by "Python" if "Python" is buried in a paragraph of text. Extraction turns unstructured text into structured data you can query, filter, and analyze.

For our job market analyzer, extraction identifies skills ("Python", "machine learning"), locations ("Amsterdam", "remote"), experience levels ("5+ years"), and more - making the data actually usable.

## The Key Ideas

### Named Entity Recognition (NER)

NER identifies entities like people, organizations, locations in text:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Google is hiring in Amsterdam.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Google ORG
# Amsterdam GPE
```

Good for standard entities. Less useful for domain-specific things like "TensorFlow" or "Senior level."

### Custom Entity Patterns

You can extend NER with your own patterns:

```python
from spacy.matcher import PhraseMatcher

skills = ["Python", "TensorFlow", "AWS", "Docker"]
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(skill) for skill in skills]
matcher.add("SKILL", patterns)
```

This catches domain terms that standard NER misses.

### Keyword Extraction

TF-IDF finds words that are important in a document but rare overall:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
# High TF-IDF = important to this document
```

KeyBERT uses embeddings for semantic keyword extraction:

```python
from keybert import KeyBERT
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(text, top_n=5)
```

### Regex Patterns

For well-defined patterns, regex is fast and precise:

```python
import re

email_pattern = r'\b[\w.-]+@[\w.-]+\.\w+\b'
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
url_pattern = r'https?://\S+'

emails = re.findall(email_pattern, text)
```

Use regex for emails, phones, URLs - structured things with predictable formats.

### LLM-Based Extraction

When you need flexibility and context understanding, use the LLM:

```python
class JobInfo(BaseModel):
    skills: list[str]
    experience_years: int | None
    location: str | None
    remote: bool

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Extract info: {job_text}"}],
    response_format=JobInfo
)
```

LLM extraction understands context ("3-5 years experience" vs "5 years ago") but is slower and more expensive.

### Fuzzy Matching

Real data has typos: "Tensorflow", "TensorFlow", "tensor flow". Fuzzy matching handles this:

```python
from rapidfuzz import fuzz, process

# Find the closest match
match = process.extractOne("Tensorflow", ["TensorFlow", "PyTorch", "Keras"])
# ("TensorFlow", 95.0, 0)  # 95% match
```

Essential for normalizing extracted entities.

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_ner_spacy.py | Standard NER with spaCy |
| 02_ner_custom.py | Add custom entity patterns |
| 03_pos_tagging.py | Part-of-speech tagging |
| 04_grammar_noun_chunks.py | Extract noun phrases |
| 05_keywords_tfidf.py | TF-IDF keyword extraction |
| 06_keywords_keybert.py | Embedding-based keywords |
| 07_lexicon_approaches.py | Dictionary-based extraction |
| 08_extraction_llm.py | LLM-based flexible extraction |
| 09_regex_patterns.py | Pattern matching with regex |
| 10_fuzzy_matching.py | Handling typos and variations |

## Choosing the Right Tool

| Method | Speed | Flexibility | Use Case |
|--------|-------|-------------|----------|
| Regex | Fast | Low | Emails, phones, URLs |
| spaCy NER | Fast | Medium | Standard entities |
| Custom patterns | Fast | High | Domain entities |
| TF-IDF | Fast | Low | Keyword discovery |
| KeyBERT | Medium | Medium | Semantic keywords |
| LLM | Slow | Very high | Complex extraction |

Start simple. Use regex and NER first. Reach for LLM extraction when you need the flexibility.

## Things to Think About

- **How do you handle extraction errors?** Not every extraction will be perfect. Build in validation and fallbacks.
- **When is the LLM worth the cost?** For high-value extraction or complex cases. Not for every job posting.
- **How do you normalize extracted data?** "Python" and "python" are the same skill. Fuzzy matching and canonicalization help.

## Related

- [Text Preparation](./text-preparation.md) - Clean text before extracting
- [Structured Output](../phase-1-foundations/structured-output.md) - LLM extraction basics
- [Classification & Routing](./classification-routing.md) - Categorizing extracted data

## Book References

- NLP_cook.5 - Information extraction
- speach_lang.III.17 - Named entity recognition
- AI_eng.5 - LLM-based extraction
