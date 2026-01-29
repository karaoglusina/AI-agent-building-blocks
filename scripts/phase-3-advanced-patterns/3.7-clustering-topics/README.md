# Module 3.7: Clustering & Topics

> *"Explore and organize large text collections"*

This module covers unsupervised learning techniques to discover patterns, group similar documents, and extract topics from large text collections without manual labeling.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_kmeans_clustering.py` | K-Means Clustering | Group documents by similarity in embedding space |
| `02_umap_visualization.py` | UMAP Visualization | Reduce high-dimensional embeddings to 2D/3D for plotting |
| `03_bertopic_basics.py` | BERTopic Basics | Automatic topic discovery with coherent themes |
| `04_cluster_labeling.py` | Cluster Labeling | Use LLM to generate human-readable cluster names |
| `05_topic_coherence.py` | Topic Coherence | Evaluate topic quality with coherence metrics |
| `06_interactive_exploration.py` | Interactive Exploration | Programmatically explore clusters for insights |

## Why Clustering & Topics?

Unsupervised learning helps you:
- **Discover patterns** in large document collections
- **Organize content** without manual labeling
- **Find market segments** automatically
- **Reduce data** to manageable themes
- **Identify outliers** and unusual cases

## Core Techniques

### 1. K-Means Clustering
```
Documents → Embeddings → K-Means → Clusters
```
Groups documents into K clusters based on similarity.

**Pros**: Simple, fast, well-understood
**Cons**: Must choose K, assumes spherical clusters

### 2. UMAP Visualization
```
High-dimensional embeddings → UMAP → 2D points → Plot
```
Reduces 384D (or higher) embeddings to 2D while preserving structure.

**Better than PCA/t-SNE**: Preserves both local and global structure

### 3. BERTopic
```
Documents → Embeddings → UMAP → HDBSCAN → c-TF-IDF → Topics
```
End-to-end topic modeling with interpretable topics.

**Advantages**: No need to specify number of topics, produces coherent themes

### 4. LLM Labeling
```
Cluster → Sample docs → LLM analysis → Human-readable label
```
Converts "Cluster 3" into "Senior Engineering Roles"

### 5. Coherence Metrics
```
Topic words → Word embeddings → Pairwise similarity → Coherence score
```
Measures how semantically related topic words are.

**Higher coherence = better topics**

## Practical Applications

### Job Market Analysis
```python
jobs = load_jobs(10000)
topics = discover_topics(jobs)
# Result: ["Entry-level Frontend", "Senior Backend", "Data Science", ...]
```

### Content Organization
```python
docs = get_all_documents()
clusters = kmeans_cluster(docs, k=20)
labels = llm_label_clusters(clusters)
# Automatic taxonomy generation!
```

### Anomaly Detection
```python
explorer = ClusterExplorer(documents)
outliers = explorer.find_outliers(cluster_id=0)
# Find unusual documents
```

## Choosing the Right Technique

| Use Case | Best Approach |
|----------|---------------|
| Quick exploration | K-Means + UMAP viz |
| Topic discovery | BERTopic |
| Known # of categories | K-Means with K |
| Unknown # of categories | BERTopic (uses HDBSCAN) |
| Need human-readable names | LLM labeling |
| Evaluate quality | Coherence metrics |
| Deep investigation | Interactive exploration |

## Prerequisites

Install required libraries:

```bash
pip install scikit-learn sentence-transformers bertopic umap-learn matplotlib
```

## Running the Scripts

Each script is self-contained:

```bash
python 01_kmeans_clustering.py
python 02_umap_visualization.py  # Creates visualization image
python 03_bertopic_basics.py
python 04_cluster_labeling.py    # Requires OPENAI_API_KEY
python 05_topic_coherence.py
python 06_interactive_exploration.py
```

## Key Insights

1. **Embeddings enable clustering** - Semantic similarity in vector space
2. **Visualization helps interpretation** - UMAP makes high-dim data visible
3. **Topics emerge naturally** - BERTopic finds themes without supervision
4. **LLMs bridge the gap** - Convert clusters to human-readable labels
5. **Quality matters** - Use coherence to tune your models
6. **Exploration reveals insights** - Go beyond summary statistics

## Parameter Tuning

### K-Means
- **n_clusters**: Use elbow method to find optimal K
- **n_init**: Run multiple times, pick best (default: 10)
- **random_state**: Set for reproducibility

### UMAP
- **n_neighbors**: 5-50, higher = preserves more global structure
- **min_dist**: 0.0-1.0, lower = tighter clusters
- **metric**: 'cosine' works well for text embeddings

### BERTopic
- **min_topic_size**: 10-50, higher = fewer but larger topics
- **nr_topics**: None (auto) or specific number
- **embedding_model**: Use domain-specific embeddings if available

## Common Patterns

### Pattern 1: Cluster → Visualize → Label
```python
clusters = kmeans(embeddings, k=5)
plot_2d = umap_visualize(embeddings, clusters)
labels = llm_label(clusters)
```

### Pattern 2: Auto-discover → Evaluate → Refine
```python
topics = bertopic(docs, min_size=10)
coherence = evaluate_coherence(topics)
if coherence < 0.4:
    topics = bertopic(docs, min_size=20)  # Try larger
```

### Pattern 3: Cluster → Explore → Insights
```python
explorer = ClusterExplorer(docs)
central = explorer.find_central_jobs(cluster_id)
outliers = explorer.find_outliers(cluster_id)
```

## Job Data Applications

Perfect for:
- **Market segmentation**: Entry-level vs Senior vs Lead roles
- **Skill clustering**: Frontend vs Backend vs Full-stack
- **Location patterns**: Remote vs Onsite vs Hybrid
- **Industry sectors**: Fintech vs Healthtech vs E-commerce
- **Salary bands**: Junior vs Mid vs Senior compensation

## Performance Considerations

| Operation | Time Complexity | Scalability |
|-----------|-----------------|-------------|
| Embeddings | O(n) | Batch: 1000s/sec |
| K-Means | O(n·k·i·d) | 10k docs: <1 min |
| UMAP | O(n log n) | 10k docs: 1-2 min |
| BERTopic | O(n log n) | 10k docs: 2-5 min |
| Coherence | O(w²) per topic | Fast (w=10-20) |

## Book References

- `hands_on_LLM.II.5` - Clustering and topic modeling with transformers
- `hands_on_LLM.III.10` - Embedding evaluation and domain adaptation
- `NLP_cook.4` - Text clustering techniques
- `NLP_cook.6` - Topic modeling and evaluation
- `NLP_cook.7` - Interactive text exploration
- `speach_lang.I.6.9` - Dimensionality reduction

## Next Steps

After mastering clustering & topics:
- Module 3.8: Evaluation Systems
- Module 3.9: Document Processing
- Module 5.2: Custom Embeddings (for better clustering)
