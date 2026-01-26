# Classification & Routing

Not every input should be handled the same way. A question about job requirements needs different handling than a request to compare two jobs. Classification figures out what kind of input you have. Routing sends it to the right handler.

## Why This Matters

Real systems need to handle diverse inputs. "Find ML jobs" is a search query. "Compare these two roles" is an analysis request. "What's the average salary?" might need a database query. Classification lets you build specialized handlers instead of one monolithic system.

For our job market analyzer, classification distinguishes between search queries, follow-up questions, comparison requests, and small talk - routing each to the appropriate handler.

## The Key Ideas

### Zero-Shot Classification

LLMs can classify without examples - just describe the categories:

```python
class QueryType(BaseModel):
    category: Literal["search", "compare", "analyze", "clarify", "other"]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{
        "role": "system",
        "content": "Classify the user query into one of these categories: search, compare, analyze, clarify, other"
    }, {
        "role": "user",
        "content": user_query
    }],
    response_format=QueryType
)
```

Good for prototyping. Fast to set up, no training data needed.

### Few-Shot Classification

Examples improve consistency:

```python
messages = [
    {"role": "system", "content": "Classify queries into: search, compare, analyze, other"},
    {"role": "user", "content": "Find Python jobs in Berlin"},
    {"role": "assistant", "content": '{"category": "search"}'},
    {"role": "user", "content": "How does job A compare to job B?"},
    {"role": "assistant", "content": '{"category": "compare"}'},
    {"role": "user", "content": user_query}  # New query
]
```

More reliable than zero-shot, especially for subtle distinctions.

### SetFit: Few-Shot Training

When you need speed at inference time, train a small model:

```python
from setfit import SetFitModel, SetFitTrainer

# Just 8-16 examples per class
train_data = [
    ("Find ML jobs", "search"),
    ("Compare these roles", "compare"),
    # ...
]

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
trainer = SetFitTrainer(model=model, train_dataset=train_data)
trainer.train()

# Fast inference
label = model.predict(["Find data science positions"])
```

Train once, classify fast. Good for production with high volume.

### Intent Detection

Intent detection is classification applied to user queries:

```python
class Intent(BaseModel):
    intent: Literal["find_jobs", "compare_jobs", "get_details", "set_preference", "unknown"]
    confidence: float
    entities: dict  # extracted parameters

# "Show me remote ML jobs" -> intent: find_jobs, entities: {"remote": True, "role": "ML"}
```

Extract not just the intent but the parameters too.

### Query Routing

Once you know the intent, route to the right handler:

```python
def handle_query(query: str):
    intent = detect_intent(query)

    match intent.intent:
        case "find_jobs":
            return search_handler(query, intent.entities)
        case "compare_jobs":
            return comparison_handler(query, intent.entities)
        case "get_details":
            return detail_handler(query, intent.entities)
        case _:
            return general_handler(query)
```

Each handler is specialized. Search queries go to the vector database. Comparisons go to an analysis pipeline.

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_zero_shot.py | LLM classification without training data |
| 02_few_shot.py | Classification with examples |
| 03_setfit_fewshot.py | Train a classifier with minimal data |
| 04_intent_detection.py | Detect user intent from queries |
| 05_query_routing.py | Route queries to specialized handlers |
| 06_sentiment.py | Classify emotional tone |

## Choosing the Right Approach

| Method | Training Data | Inference Speed | Accuracy |
|--------|---------------|-----------------|----------|
| Zero-shot LLM | None | Slow | Good |
| Few-shot LLM | 3-5 examples | Slow | Better |
| SetFit | 8-16 per class | Fast | Good |
| Fine-tuned | 100s+ examples | Very fast | Best |

Start with zero-shot LLM. Move to few-shot for consistency. Use SetFit when latency matters.

## Things to Think About

- **What happens when classification is wrong?** Build fallbacks. "I'm not sure what you mean" is better than wrong behavior.
- **How do you handle ambiguous queries?** Some queries genuinely fit multiple categories. Return confidence scores and handle low-confidence cases specially.
- **When do you need a confidence threshold?** High-stakes routing should require high confidence. Low confidence → ask for clarification.

## Related

- [Agent Orchestration](./agent-orchestration.md) - Using classification in agent routing
- [Structured Output](../phase-1-foundations/structured-output.md) - Getting structured classification results
- [Evaluation Basics](./evaluation-basics.md) - Measuring classification accuracy

## Book References

- AI_eng.5 - Prompt engineering for classification
- hands_on_LLM.II.6 - Few-shot learning
- hands_on_LLM.III.11 - SetFit
- speach_lang.II.15.3 - Intent detection
