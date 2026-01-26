# Module 5.3: Advanced NLP

> *"Extract structured information from unstructured text"*

This module covers advanced NLP techniques for extracting structured information - dependency parsing for grammatical structure, relation extraction for entity relationships, and coreference resolution for reference tracking.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_dependency_parsing.py` | Dependency Parsing | Extract grammatical structure with spaCy |
| `02_relation_extraction.py` | Relation Extraction | Extract relationships between entities |
| `03_coreference.py` | Coreference Resolution | Resolve pronouns to their antecedents |

## Why Advanced NLP?

While LLMs are powerful, traditional NLP techniques offer:
- **Precision**: Rule-based extraction for specific patterns
- **Speed**: Much faster than LLM calls
- **Cost**: No API costs, run locally
- **Interpretability**: Understand exactly what was extracted
- **Control**: Fine-tune for your specific needs

## Core Concepts

### 1. Dependency Parsing

Reveals grammatical relationships between words:

```
Sentence: "The cat chased the mouse"

Dependencies:
  chased (ROOT)
  ├── cat (subject)
  │   └── The (determiner)
  └── mouse (object)
      └── the (determiner)
```

**Use cases:**
- Extract subject-verb-object triples
- Find modifiers and attributes
- Understand sentence structure
- Extract facts from text

**Common dependencies:**
- `nsubj`: Nominal subject
- `dobj`: Direct object
- `amod`: Adjectival modifier
- `prep`: Prepositional modifier
- `ROOT`: Root of sentence

### 2. Relation Extraction

Identifies semantic relationships between entities:

```
Text: "Steve Jobs founded Apple in 1976"

Entities:
- Steve Jobs (PERSON)
- Apple (ORG)
- 1976 (DATE)

Relations:
- (Steve Jobs, FOUNDED, Apple)
- (Apple, FOUNDED_IN, 1976)
```

**Approaches:**
- **Rule-based**: Pattern matching (high precision)
- **Supervised**: Train on labeled examples
- **Distant supervision**: Auto-label from knowledge base
- **LLM-based**: Zero-shot with prompts

**Applications:**
- Build knowledge graphs
- Extract facts from documents
- Answer complex queries
- Discover entity connections

### 3. Coreference Resolution

Links expressions that refer to the same entity:

```
Text: "Barack Obama was born in Hawaii. He served as president."

Coreferences:
- "Barack Obama" = "He"
- Both refer to same person
```

**Types:**
- Pronouns: he, she, it, they
- Definite NPs: "the president"
- Names: "Obama" = "Barack Obama"
- Demonstratives: this, that, these

**Benefits:**
- Clearer text understanding
- Better information extraction
- Improved question answering
- Enhanced summarization

## Running the Examples

Each script demonstrates different NLP techniques:

```bash
# Dependency parsing
python modules/phase5/5.3-advanced-nlp/01_dependency_parsing.py

# Relation extraction
python modules/phase5/5.3-advanced-nlp/02_relation_extraction.py

# Coreference resolution
python modules/phase5/5.3-advanced-nlp/03_coreference.py
```

## Practical Workflows

### Workflow 1: Extract Facts from Documents

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_facts(text):
    """Extract (subject, verb, object) triples."""
    doc = nlp(text)
    facts = []

    for token in doc:
        if token.pos_ == "VERB":
            subject = None
            obj = None

            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child.text
                if child.dep_ in ["dobj", "attr"]:
                    obj = child.text

            if subject and obj:
                facts.append((subject, token.text, obj))

    return facts

# Test
text = "Tesla produces electric vehicles"
facts = extract_facts(text)
print(facts)  # [('Tesla', 'produces', 'vehicles')]
```

### Workflow 2: Build Knowledge Graph

```python
def build_knowledge_graph(documents):
    """Extract relations and build knowledge graph."""

    kg = {}  # {entity: [(relation, target), ...]}

    for doc in documents:
        # Extract entities
        entities = extract_entities(doc)

        # Extract relations
        relations = extract_relations(doc)

        # Add to graph
        for subj, pred, obj in relations:
            if subj not in kg:
                kg[subj] = []
            kg[subj].append((pred, obj))

    return kg

# Query the graph
def query_entity(kg, entity):
    """Get all relations for an entity."""
    return kg.get(entity, [])

# Example
kg = build_knowledge_graph(company_docs)
apple_facts = query_entity(kg, "Apple")
# [('FOUNDED_BY', 'Steve Jobs'), ('HEADQUARTERED_IN', 'Cupertino')]
```

### Workflow 3: Resolve References for Extraction

```python
def extract_with_coref(text):
    """Extract facts with coreference resolution."""

    # Step 1: Resolve coreferences
    resolved_text = resolve_coreferences(text)

    # Step 2: Extract facts
    facts = extract_facts(resolved_text)

    return facts

# Without coref:
text = "Apple released iPhone. It was revolutionary."
# Extracts: (It, was, revolutionary) ← unclear

# With coref:
resolved = "Apple released iPhone. iPhone was revolutionary."
# Extracts: (iPhone, was, revolutionary) ← clear!
```

## Best Practices

### 1. Dependency Parsing
- **Choose right model**: `en_core_web_sm` (fast) vs `en_core_web_lg` (accurate)
- **Batch processing**: Use `nlp.pipe()` for multiple documents
- **Disable unused pipes**: Speed up with `disable=['ner', 'textcat']`
- **Validate results**: Dependencies not perfect on complex sentences
- **Custom rules**: Add domain-specific patterns

### 2. Relation Extraction
- **Start with NER**: Good entity recognition is foundation
- **Combine approaches**: Rules + ML + LLM for coverage
- **Validate extractions**: Filter low-confidence relations
- **Handle negations**: Don't extract negated relations
- **Normalize relations**: Standardize relation names
- **Track provenance**: Record source text

### 3. Coreference Resolution
- **Clean text first**: Poor sentence splitting breaks coreference
- **Validate resolution**: Check if resolved text makes sense
- **Combine with NER**: Use entity types to filter candidates
- **LLM as backup**: Use GPT-4 for difficult cases
- **Test thoroughly**: Errors propagate to downstream tasks

## Common Pitfalls

### 1. Not Handling Complex Sentences
**Problem**: Parsers struggle with long, complex sentences
**Solution**: Split sentences, use larger models, add error handling

### 2. Missing Context
**Problem**: Extracting relations without understanding negation/context
**Solution**: Check for negation markers, use LLMs for ambiguous cases

### 3. Coreference Errors
**Problem**: Wrong antecedent resolution
**Solution**: Validate resolutions, use multiple approaches

### 4. Domain Mismatch
**Problem**: General models fail on domain-specific text
**Solution**: Fine-tune models, add custom patterns

### 5. Not Combining Techniques
**Problem**: Using single approach misses many relations
**Solution**: Combine rule-based, ML, and LLM methods

## Tools and Libraries

### Core Libraries
```bash
# spaCy for parsing and NER
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg  # More accurate

# Coreference resolution
pip install fastcoref  # Modern, recommended

# Visualization
pip install spacy[transformers]  # For visualizations
```

### Alternative Tools
- **AllenNLP**: Research-grade NLP
- **Stanza**: Stanford NLP in Python
- **neuralcoref**: spaCy coreference (archived but still works)

## Performance Comparison

### Dependency Parsing
```
Model          Speed      Accuracy   Size
────────────────────────────────────────
en_core_web_sm 50ms       Good       12MB
en_core_web_md 100ms      Better     40MB
en_core_web_lg 200ms      Best       560MB
```

### Relation Extraction
```
Method         Speed      Precision  Recall
────────────────────────────────────────────
Rules          1ms        90%        40%
Supervised ML  10ms       75%        70%
LLM (GPT-4)    500ms      85%        85%
Hybrid         15ms       85%        75%
```

### Coreference Resolution
```
Tool           Speed      F1 Score   Setup
────────────────────────────────────────────
FastCoref      50ms       78%        Easy
AllenNLP       200ms      82%        Complex
LLM (GPT-4)    1000ms     90%        Easiest
```

## Real-World Examples

### Example 1: Financial News Analysis
```
Task: Extract company-event relationships from news
Approach: Dependency parsing + rule-based relation extraction
Text: "Apple announced record profits for Q4"
Extracted: (Apple, ANNOUNCED, profits), (profits, IN, Q4)
Results: 85% precision, 72% recall
```

### Example 2: Biomedical Literature Mining
```
Task: Extract drug-disease relationships from papers
Approach: NER + supervised relation extraction
Text: "Aspirin reduces risk of heart attack"
Extracted: (Aspirin, REDUCES_RISK_OF, heart attack)
Results: 10K relations extracted from 1K papers
```

### Example 3: Customer Support Tickets
```
Task: Extract product-problem relationships
Approach: Dependency parsing + coreference + LLM
Text: "My iPhone won't charge. It just stopped working."
Resolved: "My iPhone won't charge. iPhone just stopped working."
Extracted: (iPhone, HAS_PROBLEM, won't charge), (iPhone, STOPPED, working)
```

## Integration with LLMs

### Hybrid Approach
Use traditional NLP for structure, LLMs for semantics:

```python
def hybrid_extraction(text):
    """Combine spaCy and LLM."""

    # 1. Fast: Use spaCy for basic structure
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # 2. If uncertain, use LLM
    if confidence_low(entities):
        entities = llm_extract_entities(text)

    # 3. Extract relations with rules
    relations = rule_based_relations(doc, entities)

    # 4. Validate with LLM
    validated = llm_validate_relations(relations)

    return validated
```

## Book References

- `NLP_cook.2` - Dependency parsing techniques
- `speach_lang.III.19` - Syntax and dependency grammar
- `speach_lang.III.20` - Relation extraction methods
- `speach_lang.III.23` - Coreference resolution

## Next Steps

After mastering advanced NLP:
- Module 5.1: Fine-tuning LLMs - Fine-tune models for NLP tasks
- Module 5.2: Custom Embeddings - Better semantic understanding
- Module 5.4: Multimodal - Extract from images and documents
- Module 3.3: Knowledge Graphs - Build graphs from extracted relations

## Additional Resources

- **spaCy Docs**: https://spacy.io/
- **spaCy Course**: https://course.spacy.io/
- **Stanford NLP**: https://nlp.stanford.edu/
- **Papers**:
  - Dependency Parsing: https://arxiv.org/abs/1412.7449
  - Relation Extraction: https://arxiv.org/abs/1906.03158
  - Coreference: https://arxiv.org/abs/1907.10529
