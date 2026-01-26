# Module 2.3: Classification & Routing

> *"Categorize content and route to appropriate handlers"*

This module covers text classification and query routing patterns essential for building intelligent systems.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_zero_shot.py` | Zero-Shot Classification | LLMs can classify without training data |
| `02_few_shot.py` | Few-Shot Classification | Examples guide behavior more reliably than instructions |
| `03_setfit_fewshot.py` | SetFit Few-Shot Training | Train a classifier with just 8-16 examples per class |
| `04_intent_detection.py` | Intent Detection | Route user requests to the right handler |
| `05_query_routing.py` | Query Router | Separate concerns with specialized handlers |
| `06_sentiment.py` | Sentiment Analysis | Reveal emotional tone in text |

## Job Data Application

- Classify jobs by category, seniority, and technical requirements
- Detect query intent ("find jobs", "compare", "summarize")
- Route queries to appropriate backends (vector search, SQL, chat)
- Analyze job description sentiment for culture insights

## Prerequisites

Install the required libraries:

```bash
pip install openai pydantic setfit sentence-transformers textblob
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_zero_shot.py
python 02_few_shot.py
# ... etc
```

## Classification Method Comparison

| Method | Training Data | Speed | Accuracy | Use Case |
|--------|---------------|-------|----------|----------|
| Zero-shot LLM | None | Slow | Good | Quick prototyping |
| Few-shot LLM | 3-5 examples | Slow | Better | Consistent formatting |
| SetFit | 8-16 per class | Fast (after) | Good | Production at scale |
| Fine-tuned | 100s+ examples | Fast | Best | High-volume production |

## Book References

- `AI_eng.5` - Prompt engineering for classification
- `hands_on_LLM.II.6` - Few-shot learning
- `hands_on_LLM.III.11` - SetFit and efficient training
- `speach_lang.II.15.3` - Intent detection
- `NLP_cook.8` - Text classification
