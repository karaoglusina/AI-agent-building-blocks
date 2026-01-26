"""
01 - Local Embeddings with Sentence Transformers
=================================================
Use sentence-transformers models for local embedding generation.

Key concept: Sentence Transformers provides pre-trained models for generating high-quality embeddings locally, without API calls.

Book reference: hands_on_LLM.I.2, hands_on_LLM.III.10
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import numpy as np
from typing import List


def sentence_transformers_intro():
    """Introduce Sentence Transformers library."""
    print("=== SENTENCE TRANSFORMERS ===\n")

    print("What is Sentence Transformers?")
    print("  A Python library for state-of-the-art sentence, text, and image embeddings.")
    print("  Built on top of Hugging Face Transformers.\n")

    print("Why use Sentence Transformers?")
    print("  ✓ Run embeddings locally (no API costs)")
    print("  ✓ Privacy - data never leaves your machine")
    print("  ✓ Fast - optimized for batch processing")
    print("  ✓ Many pre-trained models to choose from")
    print("  ✓ Easy to fine-tune for your domain")
    print("  ✓ Supports semantic search, clustering, etc.\n")

    print("Installation:")
    print("  pip install sentence-transformers")


def popular_models():
    """Show popular Sentence Transformer models."""
    print("\n" + "=" * 70)
    print("=== POPULAR MODELS ===\n")

    models = [
        ("all-MiniLM-L6-v2",
         "Fast, 384 dim",
         "General purpose, good balance speed/quality",
         "22M params"),

        ("all-mpnet-base-v2",
         "High quality, 768 dim",
         "Best overall quality for most tasks",
         "110M params"),

        ("multi-qa-MiniLM-L6-cos-v1",
         "Fast, 384 dim",
         "Optimized for question-answering",
         "22M params"),

        ("paraphrase-multilingual-MiniLM-L12-v2",
         "Multilingual, 384 dim",
         "Supports 50+ languages",
         "118M params"),

        ("msmarco-distilbert-base-v4",
         "Fast, 768 dim",
         "Trained on MS MARCO search dataset",
         "66M params"),
    ]

    print("Model                              Specs              Use Case                    Size")
    print("-" * 100)
    for name, specs, use_case, size in models:
        print(f"{name:35} {specs:18} {use_case:27} {size}")

    print("\n\nChoosing a model:")
    print("  Speed priority → all-MiniLM-L6-v2")
    print("  Quality priority → all-mpnet-base-v2")
    print("  Question-answering → multi-qa-MiniLM-L6-cos-v1")
    print("  Multilingual → paraphrase-multilingual-MiniLM-L12-v2")


def basic_usage_example():
    """Show basic usage example (conceptual)."""
    print("\n" + "=" * 70)
    print("=== BASIC USAGE EXAMPLE ===\n")

    code = '''from sentence_transformers import SentenceTransformer

# 1. Load model (downloads on first use)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on the rug",
    "The dog plays in the yard"
]

embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")  # (3, 384)
print(f"Type: {type(embeddings)}")   # numpy.ndarray

# 3. Compute similarity
from sentence_transformers.util import cos_sim

similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.3f}")  # High (same meaning)

similarity = cos_sim(embeddings[0], embeddings[2])
print(f"Similarity: {similarity.item():.3f}")  # Low (different meaning)
'''

    print(code)


def semantic_search_example():
    """Show semantic search example."""
    print("\n" + "=" * 70)
    print("=== SEMANTIC SEARCH EXAMPLE ===\n")

    print("Scenario: Search through a document collection\n")

    code = '''from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Document collection
documents = [
    "Python is a high-level programming language",
    "Machine learning is a subset of AI",
    "The Eiffel Tower is located in Paris",
    "Neural networks are inspired by the brain",
    "JavaScript is used for web development"
]

# Encode documents once
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Search query
query = "What is artificial intelligence?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Find most similar documents
similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
top_results = similarities.argsort(descending=True)[:3]

print("Query:", query)
print("\\nTop results:")
for idx in top_results:
    print(f"  [{similarities[idx]:.3f}] {documents[idx]}")

# Output:
# [0.524] Machine learning is a subset of AI
# [0.487] Neural networks are inspired by the brain
# [0.312] Python is a high-level programming language
'''

    print(code)


def clustering_example():
    """Show clustering example."""
    print("\n" + "=" * 70)
    print("=== CLUSTERING EXAMPLE ===\n")

    print("Scenario: Group similar sentences together\n")

    code = '''from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love pizza",
    "Pizza is my favorite food",
    "The weather is nice today",
    "It's sunny outside",
    "Python is great for data science",
    "I use Python for machine learning"
]

# Encode
embeddings = model.encode(sentences)

# Cluster (3 groups)
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(embeddings)

# Group by cluster
from collections import defaultdict
clusters = defaultdict(list)
for sent, label in zip(sentences, labels):
    clusters[label].append(sent)

print("Clusters:")
for cluster_id, sents in clusters.items():
    print(f"\\nCluster {cluster_id}:")
    for sent in sents:
        print(f"  - {sent}")

# Output:
# Cluster 0: (Food)
#   - I love pizza
#   - Pizza is my favorite food
# Cluster 1: (Weather)
#   - The weather is nice today
#   - It's sunny outside
# Cluster 2: (Python)
#   - Python is great for data science
#   - I use Python for machine learning
'''

    print(code)


def performance_optimization():
    """Show performance optimization techniques."""
    print("\n" + "=" * 70)
    print("=== PERFORMANCE OPTIMIZATION ===\n")

    print("1. Batch encoding")
    print("   Encode multiple sentences at once for speed\n")

    code1 = '''# Slow: One at a time
for sentence in sentences:
    embedding = model.encode(sentence)

# Fast: Batch encoding
embeddings = model.encode(sentences, batch_size=32)
'''
    print(code1)

    print("\n2. Use GPU")
    print("   10-100× faster with CUDA\n")

    code2 = '''# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Model automatically uses GPU if available
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Device: {model.device}")

# Force CPU (for testing)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
'''
    print(code2)

    print("\n3. Convert to tensor for faster similarity")
    print("   PyTorch tensors are faster than NumPy\n")

    code3 = '''# Returns PyTorch tensor (faster for similarity)
embeddings = model.encode(sentences, convert_to_tensor=True)

# Use util.cos_sim for batch similarity
similarities = util.cos_sim(embeddings, embeddings)
'''
    print(code3)

    print("\n4. Normalize embeddings")
    print("   Pre-normalize for dot product similarity\n")

    code4 = '''# Encode with normalization
embeddings = model.encode(sentences, normalize_embeddings=True)

# Now dot product = cosine similarity
similarity = np.dot(embeddings[0], embeddings[1])
'''
    print(code4)


def comparison_with_openai():
    """Compare with OpenAI embeddings."""
    print("\n" + "=" * 70)
    print("=== COMPARISON WITH OPENAI ===\n")

    print("Aspect              OpenAI (text-embedding-3-small)    Sentence Transformers")
    print("-" * 85)
    print("Dimensions          1536                                384-768 (model dependent)")
    print("Cost                $0.02 / 1M tokens                   Free (one-time GPU cost)")
    print("Latency             ~50-200ms (API)                     ~5-50ms (local)")
    print("Privacy             Data sent to OpenAI                 Fully local")
    print("Quality             Excellent                           Very Good to Excellent")
    print("Internet required   Yes                                 No (after download)")
    print("Rate limits         Yes (500 req/min)                   No")
    print("-" * 85)

    print("\n\nWhen to use each:\n")

    print("OpenAI:")
    print("  • Need absolute best quality")
    print("  • Don't have GPU")
    print("  • Small scale (<100K embeddings)")
    print("  • Rapid prototyping\n")

    print("Sentence Transformers:")
    print("  • Privacy concerns (healthcare, legal)")
    print("  • Large scale (millions of embeddings)")
    print("  • Cost sensitive")
    print("  • Need low latency")
    print("  • Want to fine-tune on your domain")


def model_selection_guide():
    """Guide for selecting the right model."""
    print("\n" + "=" * 70)
    print("=== MODEL SELECTION GUIDE ===\n")

    print("By use case:\n")

    use_cases = [
        ("General semantic search", "all-mpnet-base-v2"),
        ("Fast semantic search", "all-MiniLM-L6-v2"),
        ("Question answering", "multi-qa-MiniLM-L6-cos-v1"),
        ("Code search", "code-search-net-corpus"),
        ("Duplicate detection", "paraphrase-MiniLM-L6-v2"),
        ("Multilingual", "paraphrase-multilingual-MiniLM-L12-v2"),
        ("Scientific papers", "allenai-specter"),
    ]

    for use_case, model in use_cases:
        print(f"  {use_case:25} → {model}")

    print("\n\nBy constraint:\n")

    constraints = [
        ("Speed critical", "MiniLM models (384 dim, fast)"),
        ("Quality critical", "MPNet models (768 dim, slow)"),
        ("Memory limited", "TinyBERT models (128 dim)"),
        ("GPU limited", "Small models (MiniLM)"),
    ]

    for constraint, recommendation in constraints:
        print(f"  {constraint:20} → {recommendation}")


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Choose right model", "Match model to your task (QA, semantic search, etc.)"),
        ("Batch processing", "Encode in batches for speed"),
        ("Use GPU", "10-100× faster with CUDA"),
        ("Normalize embeddings", "For cosine similarity via dot product"),
        ("Cache embeddings", "Don't re-encode same text"),
        ("Monitor memory", "Large batches can OOM"),
        ("Evaluate on your data", "Test model quality on your specific domain"),
        ("Consider fine-tuning", "For domain-specific improvement"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def common_pitfalls():
    """Show common pitfalls."""
    print("=" * 70)
    print("=== COMMON PITFALLS ===\n")

    pitfalls = [
        ("Using wrong similarity metric",
         "Use cosine similarity for sentence embeddings, not Euclidean"),

        ("Not normalizing for dot product",
         "Must normalize embeddings if using dot product for similarity"),

        ("Encoding one at a time",
         "Batch encoding is much faster"),

        ("Mixing models",
         "Don't compare embeddings from different models"),

        ("Ignoring max sequence length",
         "Models have max length (often 512 tokens) - truncates longer text"),

        ("Not caching embeddings",
         "Re-encoding same text wastes computation"),
    ]

    for pitfall, solution in pitfalls:
        print(f"✗ {pitfall}")
        print(f"  → {solution}\n")


def practical_tips():
    """Provide practical tips."""
    print("=" * 70)
    print("=== PRACTICAL TIPS ===\n")

    tips = [
        "1. Start with all-MiniLM-L6-v2 (fast, good quality)",
        "2. Upgrade to all-mpnet-base-v2 if quality insufficient",
        "3. Encode documents once, store embeddings",
        "4. Use convert_to_tensor=True for similarity computations",
        "5. Set batch_size based on GPU memory (32-128 typical)",
        "6. Monitor encoding speed (should be >100 sentences/sec on GPU)",
        "7. Test model on your data before committing",
        "8. Consider fine-tuning if off-the-shelf models don't work well",
        "9. Use show_progress_bar=True for long encoding jobs",
        "10. Save embeddings in efficient format (numpy .npz or faiss index)",
    ]

    for tip in tips:
        print(f"  {tip}")


if __name__ == "__main__":
    sentence_transformers_intro()
    popular_models()
    basic_usage_example()
    semantic_search_example()
    clustering_example()
    performance_optimization()
    comparison_with_openai()
    model_selection_guide()
    best_practices()
    common_pitfalls()
    practical_tips()

    print("\n" + "=" * 70)
    print("\nKey insight: Sentence Transformers makes embeddings accessible!")
    print("Run high-quality embeddings locally, privately, and for free")
