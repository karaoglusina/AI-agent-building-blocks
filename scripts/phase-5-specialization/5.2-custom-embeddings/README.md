# Module 5.2: Custom Embeddings

> *"Generate and fine-tune embeddings for your specific domain"*

This module covers working with custom embedding models - using Sentence Transformers for local embeddings, domain adaptation with TSDAE, evaluating embedding quality, and understanding bias in embeddings.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_sentence_transformers.py` | Local Embeddings | Use sentence-transformers models locally |
| `02_domain_adaptation_tsdae.py` | Domain Adaptation with TSDAE | Unsupervised domain adaptation for embeddings |
| `03_embedding_evaluation.py` | Embedding Evaluation | Measure embedding quality rigorously |
| `04_bias_in_embeddings.py` | Bias in Embeddings | Awareness and mitigation of embedding biases |

## Why Custom Embeddings?

While OpenAI and other API-based embeddings are excellent, custom embeddings offer:
- **Privacy**: Data never leaves your infrastructure
- **Cost**: No per-token API costs after initial setup
- **Latency**: Local inference is faster than API calls
- **Customization**: Fine-tune for your specific domain
- **Control**: Own your embedding infrastructure

## Core Concepts

### 1. Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Hello world", "Goodbye world"]
embeddings = model.encode(texts)

# embeddings.shape: (2, 384)
```

**Popular models:**
- `all-MiniLM-L6-v2`: Fast, 384 dimensions (recommended start)
- `all-mpnet-base-v2`: High quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

### 2. Domain Adaptation with TSDAE

TSDAE (Transformer Denoising Auto-Encoder) adapts embeddings to your domain without labeled data:

```
Input:  "Machine learning is a subset of AI"
Noise:  "Machine _____ is _____ subset _____ AI"  (60% deletion)
Train:  Model learns to reconstruct original from noisy version
Result: Better embeddings for your domain vocabulary
```

**When to use:**
- Medical, legal, or technical domain text
- Off-the-shelf models struggle on your data
- Have unlabeled domain text (1K-100K sentences)
- Need better semantic search in your domain

### 3. Evaluation Metrics

**Semantic Similarity:**
- Spearman correlation with human judgments
- Tests if similar texts have similar embeddings

**Retrieval:**
- MRR (Mean Reciprocal Rank): Position of first relevant result
- Recall@K: Relevant documents in top K
- MAP (Mean Average Precision): Overall retrieval quality

**Clustering:**
- Silhouette Score: How well-separated clusters are
- ARI (Adjusted Rand Index): Match with ground truth

### 4. Bias Awareness

Embeddings encode biases from training data:

```
Gender bias:    'doctor' closer to 'man' than 'woman'
Racial bias:    Names treated differently by ethnicity
Age bias:       'young' associated with 'innovative'
```

**Mitigation approaches:**
- Balanced training data
- Post-processing debiasing
- Regular bias audits
- Transparency about limitations

## Running the Examples

Each script demonstrates different aspects of custom embeddings:

```bash
# Sentence Transformers basics
python "scripts/phase-5-specialization/5.2-custom-embeddings/01_sentence_transformers.py"
# Domain adaptation with TSDAE
python "scripts/phase-5-specialization/5.2-custom-embeddings/02_domain_adaptation_tsdae.py"
# Evaluating embedding quality
python "scripts/phase-5-specialization/5.2-custom-embeddings/03_embedding_evaluation.py"
# Understanding and measuring bias
python "scripts/phase-5-specialization/5.2-custom-embeddings/04_bias_in_embeddings.py"
```

## Practical Workflows

### Workflow 1: Using Off-the-Shelf Models

```python
from sentence_transformers import SentenceTransformer, util

# 1. Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Encode your documents
documents = ["doc1...", "doc2...", ...]
doc_embeddings = model.encode(documents)

# 3. Search
query = "search query"
query_emb = model.encode(query)
similarities = util.cos_sim(query_emb, doc_embeddings)[0]

# 4. Get top results
top_indices = similarities.argsort(descending=True)[:5]
for idx in top_indices:
    print(f"{documents[idx]} ({similarities[idx]:.3f})")
```

### Workflow 2: Domain Adaptation

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset

# 1. Collect domain text
domain_texts = load_domain_documents()  # 10K+ sentences

# 2. Load base model
model = SentenceTransformer('all-mpnet-base-v2')

# 3. Create denoising dataset
train_dataset = DenoisingAutoEncoderDataset(domain_texts)

# 4. Train with TSDAE
train_loss = losses.DenoisingAutoEncoderLoss(model)
model.fit([(train_dataloader, train_loss)], epochs=1)

# 5. Save adapted model
model.save('models/domain-adapted')
```

### Workflow 3: Evaluation

```python
# Prepare test data
test_queries = [...]
test_docs = [...]
relevance_map = {query_idx: [relevant_doc_indices]}

# Evaluate models
models = {
    'base': SentenceTransformer('all-MiniLM-L6-v2'),
    'adapted': SentenceTransformer('./domain-adapted')
}

for name, model in models.items():
    mrr = calculate_mrr(model, test_queries, test_docs, relevance_map)
    print(f"{name}: MRR = {mrr:.3f}")
```

## Best Practices

### 1. Model Selection
- **Start simple**: `all-MiniLM-L6-v2` for prototyping
- **Quality critical**: Upgrade to `all-mpnet-base-v2`
- **Speed critical**: Stick with MiniLM models
- **Multilingual**: Use `paraphrase-multilingual-*` models
- **Domain-specific**: Check Hugging Face for specialized models

### 2. Performance Optimization
- **Batch encoding**: Encode multiple texts at once
  ```python
  embeddings = model.encode(texts, batch_size=32)
  ```
- **Use GPU**: 10-100× faster
  ```python
  model = SentenceTransformer('model-name', device='cuda')
  ```
- **Normalize embeddings**: For faster similarity
  ```python
  embeddings = model.encode(texts, normalize_embeddings=True)
  ```
- **Cache embeddings**: Don't re-encode same text

### 3. Domain Adaptation
- **Data quality**: Clean, domain-representative text
- **Sufficient data**: 10K+ sentences recommended
- **Evaluate improvement**: Compare before/after
- **One epoch often enough**: More may overfit
- **Keep base model strong**: Start with MPNet, not MiniLM

### 4. Evaluation
- **Use YOUR data**: Generic benchmarks may not correlate
- **Multiple metrics**: MRR, Recall@K, human evaluation
- **Baseline comparison**: Compare to existing solution
- **Statistical significance**: Multiple runs, confidence intervals
- **Monitor production**: Evaluation doesn't stop at deployment

### 5. Bias Mitigation
- **Measure regularly**: Test for known biases
- **Document limitations**: Be transparent
- **Diverse data**: Ensure representation
- **Human oversight**: Don't fully automate sensitive decisions
- **Regular audits**: Third-party fairness assessments

## Common Pitfalls

### 1. Wrong Similarity Metric
**Problem**: Using Euclidean distance instead of cosine similarity
**Solution**: Always use cosine similarity for sentence embeddings

### 2. Mixing Models
**Problem**: Comparing embeddings from different models
**Solution**: Re-encode all documents if changing models

### 3. Not Caching
**Problem**: Re-encoding same text repeatedly
**Solution**: Store embeddings in database or vector store

### 4. Ignoring Max Length
**Problem**: Models truncate text >512 tokens silently
**Solution**: Chunk long documents, or use long-context models

### 5. One-at-a-Time Encoding
**Problem**: Encoding sentences individually
**Solution**: Always batch encode for speed

### 6. No Evaluation
**Problem**: Deploying without testing on your data
**Solution**: Evaluate on representative test set before deployment

## Hardware Requirements

### Local Inference
```
Model           CPU Time    GPU Time    Memory
──────────────────────────────────────────────
MiniLM-L6       50ms        5ms         500MB
MPNet-base      150ms       15ms        1.5GB
Large models    500ms       50ms        4GB+
```

### Fine-tuning (TSDAE)
```
Dataset Size    Time (GPU)    VRAM
──────────────────────────────────
10K sentences   30 min        4GB
100K sentences  3 hours       8GB
1M sentences    30 hours      16GB
```

## Libraries

```bash
# Core
pip install sentence-transformers

# Evaluation
pip install scipy scikit-learn

# Optional (for advanced features)
pip install faiss-cpu  # Fast similarity search
pip install hnswlib    # Alternative ANN library
```

## Model Comparison

### OpenAI vs Sentence Transformers

| Aspect | OpenAI | Sentence Transformers |
|--------|--------|----------------------|
| Quality | Excellent | Very Good to Excellent |
| Dimensions | 1536 | 384-768 (model dependent) |
| Cost | $0.02/1M tokens | Free (GPU cost only) |
| Latency | 50-200ms | 5-50ms (local) |
| Privacy | Data sent to API | Fully local |
| Customization | None | Full fine-tuning control |
| Internet | Required | Not required |

**When to use OpenAI:**
- Need absolute best quality
- Don't have GPU
- Small scale (<100K embeddings)
- Rapid prototyping

**When to use Sentence Transformers:**
- Privacy concerns (healthcare, legal)
- Large scale (millions of embeddings)
- Cost sensitive
- Need low latency
- Want domain customization

## Real-World Examples

### Example 1: Legal Document Search
```
Domain: Legal contracts and case law
Approach: Domain adapt MPNet-base with 50K legal sentences
Data: Court opinions, contracts, legal briefs
Results: MRR 0.52 → 0.71 (+37%)
Training: 2 hours on single GPU
```

### Example 2: Medical Research Search
```
Domain: Biomedical research papers
Approach: Fine-tune on 100K PubMed abstracts
Data: Medical terminology, drug names, procedures
Results: Retrieval Recall@10: 0.63 → 0.82 (+30%)
Training: 4 hours on A100
```

### Example 3: Internal Company Wiki
```
Domain: Company documentation and policies
Approach: Domain adapt MiniLM with 10K sentences
Data: Company jargon, acronyms, processes
Results: User satisfaction 68% → 85% (+17%)
Training: 30 minutes on RTX 3090
```

## Advanced Topics

### 1. Multi-vector Representations
Use ColBERT for better retrieval (late interaction):
```python
# Each token gets embedding, not just sentence
# More expensive but higher quality
```

### 2. Cross-Encoders
Re-rank top results with cross-encoder:
```python
from sentence_transformers import CrossEncoder

# First stage: bi-encoder (fast)
candidates = bi_encoder_search(query, top_k=100)

# Second stage: cross-encoder (slow but accurate)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc) for doc in candidates])
final_results = top_k_by_score(candidates, scores, k=10)
```

### 3. Matryoshka Embeddings
Variable-dimension embeddings for flexibility:
```python
# Same model produces 768, 512, 256, 128, 64 dim embeddings
# Trade quality for speed/storage as needed
```

## Book References

- `hands_on_LLM.I.2` - Embedding basics
- `hands_on_LLM.III.10` - Custom embeddings and fine-tuning
- `speach_lang.I.6.11` - Bias in word embeddings
- `speach_lang.I.6.12` - Embedding evaluation

## Next Steps

After mastering custom embeddings:
- Module 5.1: Fine-tuning LLMs - Fine-tune full models
- Module 5.3: Advanced NLP - Extract structured information
- Module 5.4: Multimodal - Work with images and text
- Module 3.4: Advanced RAG - Use custom embeddings in RAG systems

## Additional Resources

- **Sentence Transformers Docs**: https://www.sbert.net/
- **Hugging Face Models**: https://huggingface.co/models?library=sentence-transformers
- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard
- **Papers**:
  - Sentence-BERT: https://arxiv.org/abs/1908.10084
  - TSDAE: https://arxiv.org/abs/2104.06979
  - Bias in Embeddings: https://arxiv.org/abs/1607.06520
