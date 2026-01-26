# Module 3.8: Evaluation Systems

> *"Production-grade evaluation"*

This module covers building systematic evaluation systems for AI applications, including test data creation, automated pipelines, regression testing, prompt versioning, cost tracking, and human evaluation design.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_eval_dataset.py` | Create Eval Dataset | Good evaluation starts with representative, well-labeled test data |
| `02_eval_pipeline.py` | Evaluation Pipeline | Automated evaluation ensures consistent quality measurement |
| `03_regression_testing.py` | Regression Testing | Compare to baseline to catch quality degradation |
| `04_prompt_versioning.py` | Prompt Versioning | Track and A/B test prompts like code |
| `05_cost_tracking.py` | Cost Tracking | Monitor and optimize API costs systematically |
| `06_human_eval_design.py` | Human Evaluation | Structure human evaluation for reliable quality assessment |

## Why Evaluation Systems?

Production AI needs rigorous evaluation:
- **Catch regressions** before deployment
- **Measure improvements** objectively
- **Track costs** and optimize spending
- **Version prompts** systematically
- **Validate quality** with humans

## Core Components

### 1. Test Dataset
- Representative samples
- Ground truth labels
- Balanced distribution
- Edge cases included

### 2. Automated Pipeline
- Run tests automatically
- Calculate metrics
- Generate reports
- Save results for comparison

### 3. Regression Detection
- Compare to baseline
- Flag degradation
- Track improvements
- Block bad deployments

### 4. Prompt Versioning
- Version control for prompts
- A/B test variants
- Measure impact
- Roll back if needed

### 5. Cost Tracking
- Token counting
- Cost estimation
- Model comparison
- Optimization opportunities

### 6. Human Evaluation
- Clear rubrics
- Blind evaluation
- Multiple annotators
- Agreement metrics

## Evaluation Workflow

```
1. Create test dataset (representative + diverse)
   ↓
2. Run baseline evaluation
   ↓
3. Save as baseline
   ↓
4. Make changes (prompt, model, code)
   ↓
5. Run evaluation again
   ↓
6. Compare to baseline
   ↓
7. If regression → investigate & fix
   If improvement → update baseline & deploy
```

## Metrics by Task Type

### Classification
- Accuracy
- Precision/Recall/F1
- Confusion matrix
- Per-class metrics

### Generation
- ROUGE/BLEU
- LLM-as-judge
- Human evaluation
- Coherence scores

### Retrieval (RAG)
- Recall@k
- Precision@k
- MRR
- NDCG

### Extraction
- Exact match
- Partial match
- F1 for fields
- Coverage

## Prerequisites

Install required libraries:

```bash
pip install openai tiktoken pydantic
```

## Running the Scripts

Run in sequence:

```bash
# 1. Create test dataset
python 01_eval_dataset.py

# 2. Run evaluation
python 02_eval_pipeline.py

# 3. Save as baseline
cp eval_results.json eval_baseline.json

# 4. Make changes to your system, then:
python 02_eval_pipeline.py

# 5. Check for regressions
python 03_regression_testing.py

# 6. Compare prompt versions
python 04_prompt_versioning.py

# 7. Analyze costs
python 05_cost_tracking.py

# 8. Design human evaluation
python 06_human_eval_design.py
```

## Key Insights

1. **Test data quality matters** - GIGO applies to evaluation
2. **Automate everything** - Manual testing doesn't scale
3. **Compare to baseline** - Absolute metrics aren't enough
4. **Version prompts** - They're code, treat them like code
5. **Track costs early** - Surprises are expensive
6. **Humans for quality** - Automated metrics miss nuances

## Cost Optimization Strategies

### Model Selection
- Use `gpt-4o-mini` for most tasks (10-20x cheaper)
- Reserve `gpt-4o` for complex/critical tasks
- Fallback strategy: try cheap model first

### Prompt Engineering
- Shorter prompts = lower costs
- Remove unnecessary context
- Use system messages wisely
- Cache common prompts

### Batching
- Process multiple items per call when possible
- Reduces overhead tokens
- Faster processing

### Token Management
- Count tokens before calling
- Truncate long inputs intelligently
- Optimize output length

## Human Evaluation Design

### When to Use Human Eval
- Qualitative assessment (tone, style, appropriateness)
- Edge cases and corner scenarios
- Overall product quality
- When automated metrics disagree with intuition

### Best Practices
1. **Clear instructions** with examples
2. **Multiple annotators** (2-3 per task)
3. **Blind evaluation** (hide system labels)
4. **Training phase** before main task
5. **Quality checks** with gold standards
6. **Measure agreement** between annotators
7. **Fair compensation** for annotators

### Annotation Tools
- Label Studio (open source)
- Prodigy (spaCy's tool)
- Scale AI (managed service)
- Custom tools (FastAPI + UI)

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Evaluation
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run evaluation
        run: python 02_eval_pipeline.py
      - name: Check regression
        run: python 03_regression_testing.py
      - name: Fail if regression
        run: |
          if [ $? -ne 0 ]; then
            exit 1
          fi
```

## Book References

- `AI_eng.3` - Evaluation metrics and methodologies
- `AI_eng.4` - Testing and evaluation systems
- `AI_eng.5` - Prompt engineering and versioning
- `AI_eng.8` - Data quality for evaluation
- `AI_eng.9` - Cost optimization strategies
- `hands_on_LLM.III.12` - Human evaluation and feedback

## Next Steps

After mastering evaluation systems:
- Module 3.9: Document Processing
- Module 4.3: Observability (Langfuse for production tracking)
- Module 4.4: Guardrails
