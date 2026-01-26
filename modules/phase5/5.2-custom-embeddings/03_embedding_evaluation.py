"""
03 - Embedding Evaluation
==========================
Measure and compare embedding quality.

Key concept: Proper evaluation ensures your embeddings capture semantic similarity well. Use multiple metrics and real-world test cases.

Book reference: hands_on_LLM.III.10, speach_lang.I.6.12
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import numpy as np
from typing import List, Tuple, Dict


def embedding_evaluation_intro():
    """Introduce embedding evaluation."""
    print("=== EMBEDDING EVALUATION ===\n")

    print("Why evaluate embeddings?")
    print("  • Verify quality before deployment")
    print("  • Compare different embedding models")
    print("  • Measure improvement after fine-tuning")
    print("  • Debug poor search/retrieval performance\n")

    print("Evaluation dimensions:")
    print("  1. Semantic similarity - Do similar texts have similar embeddings?")
    print("  2. Retrieval quality - Can we find relevant documents?")
    print("  3. Clustering quality - Are similar items grouped together?")
    print("  4. Bias and fairness - Are embeddings biased?\n")

    print("Challenges:")
    print("  • No single 'best' metric")
    print("  • Task-dependent quality")
    print("  • Need representative test data")
    print("  • Trade-offs (speed vs quality)")


def semantic_similarity_evaluation():
    """Explain semantic similarity evaluation."""
    print("\n" + "=" * 70)
    print("=== SEMANTIC SIMILARITY EVALUATION ===\n")

    print("Concept:")
    print("  Test if embeddings capture similarity between text pairs\n")

    print("Standard datasets:")
    print("  • STS Benchmark (Semantic Textual Similarity)")
    print("  • SICK (Sentences Involving Compositional Knowledge)")
    print("  • STS-B (Multi-domain similarity)\n")

    print("Evaluation metric: Spearman correlation")
    print("  Compare model similarities with human-rated similarities\n")

    print("Example test pairs:\n")

    examples = [
        ("A man is playing guitar", "A person is playing music", "High (0.8)"),
        ("A dog is running", "A cat is sleeping", "Low (0.2)"),
        ("The weather is nice", "It's sunny today", "Medium (0.6)"),
    ]

    print("  Sentence 1                 Sentence 2                  Human Rating")
    print("  " + "-" * 75)
    for sent1, sent2, rating in examples:
        print(f"  {sent1:28} {sent2:28} {rating}")

    print("\n\nCode example:\n")

    code = '''from scipy.stats import spearmanr

def evaluate_similarity(model, test_pairs):
    """
    test_pairs: [(sent1, sent2, human_score), ...]
    human_score: 0-1 (0=different, 1=identical)
    """
    human_scores = []
    model_scores = []

    for sent1, sent2, human_score in test_pairs:
        # Get embeddings
        emb1 = model.encode(sent1)
        emb2 = model.encode(sent2)

        # Compute cosine similarity
        model_score = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        human_scores.append(human_score)
        model_scores.append(model_score)

    # Spearman correlation
    correlation, p_value = spearmanr(human_scores, model_scores)
    return correlation

# Score interpretation:
# 0.9-1.0: Excellent
# 0.7-0.9: Good
# 0.5-0.7: Moderate
# <0.5: Poor
'''

    print(code)


def retrieval_evaluation():
    """Explain retrieval evaluation metrics."""
    print("\n" + "=" * 70)
    print("=== RETRIEVAL EVALUATION ===\n")

    print("Scenario: Search for relevant documents given a query\n")

    print("Common metrics:\n")

    print("1. Mean Reciprocal Rank (MRR)")
    print("   Average of 1/rank where rank is position of first relevant doc")
    print("   Formula: MRR = (1/N) × Σ(1/rank_i)")
    print("   Range: 0-1 (higher is better)")
    print("   Example: First relevant at position 3 → 1/3 = 0.333\n")

    print("2. Mean Average Precision (MAP)")
    print("   Average precision across all queries")
    print("   Considers all relevant documents, not just first")
    print("   Range: 0-1 (higher is better)\n")

    print("3. Recall@K")
    print("   Proportion of relevant docs in top K results")
    print("   Formula: Recall@10 = (relevant in top 10) / (total relevant)")
    print("   Example: 7 relevant in top 10 out of 15 total → 7/15 = 0.467\n")

    print("4. NDCG@K (Normalized Discounted Cumulative Gain)")
    print("   Accounts for position and relevance scores")
    print("   Range: 0-1 (higher is better)")
    print("   Better than Recall@K for ranked results\n")

    print("Code example:\n")

    code = '''def calculate_mrr(model, queries, documents, relevance_map):
    """
    queries: List of query strings
    documents: List of document strings
    relevance_map: {query_idx: [relevant_doc_indices]}
    """
    # Encode once
    doc_embeddings = model.encode(documents)

    reciprocal_ranks = []
    for i, query in enumerate(queries):
        query_emb = model.encode(query)

        # Calculate similarities
        similarities = np.dot(doc_embeddings, query_emb)

        # Rank documents
        ranked_indices = np.argsort(similarities)[::-1]

        # Find rank of first relevant doc
        relevant_docs = relevance_map[i]
        for rank, doc_idx in enumerate(ranked_indices, 1):
            if doc_idx in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)  # No relevant doc found

    return np.mean(reciprocal_ranks)


def calculate_recall_at_k(model, queries, documents, relevance_map, k=10):
    """Calculate Recall@K."""
    doc_embeddings = model.encode(documents)

    recalls = []
    for i, query in enumerate(queries):
        query_emb = model.encode(query)
        similarities = np.dot(doc_embeddings, query_emb)

        # Top K documents
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Count relevant docs in top K
        relevant_docs = set(relevance_map[i])
        found = len(relevant_docs.intersection(top_k_indices))
        total_relevant = len(relevant_docs)

        recalls.append(found / total_relevant if total_relevant > 0 else 0)

    return np.mean(recalls)
'''

    print(code)


def clustering_evaluation():
    """Explain clustering evaluation."""
    print("\n" + "=" * 70)
    print("=== CLUSTERING EVALUATION ===\n")

    print("Scenario: Group similar items together\n")

    print("Metrics:\n")

    print("1. Silhouette Score")
    print("   Measures how well-separated clusters are")
    print("   Range: -1 to 1 (higher is better)")
    print("   >0.7: Strong clustering")
    print("   0.5-0.7: Reasonable clustering")
    print("   <0.5: Weak clustering\n")

    print("2. Adjusted Rand Index (ARI)")
    print("   Compare clustering to ground truth labels")
    print("   Range: -1 to 1 (1 = perfect match)")
    print("   Requires true labels\n")

    print("3. V-Measure")
    print("   Harmonic mean of homogeneity and completeness")
    print("   Range: 0-1 (higher is better)")
    print("   Requires true labels\n")

    print("Code example:\n")

    code = '''from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

def evaluate_clustering(embeddings, true_labels, n_clusters):
    """Evaluate clustering quality."""

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Silhouette score (no labels needed)
    silhouette = silhouette_score(embeddings, predicted_labels)

    # ARI (requires labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    return {
        "silhouette": silhouette,
        "ari": ari
    }

# Example usage
scores = evaluate_clustering(embeddings, labels, n_clusters=5)
print(f"Silhouette: {scores['silhouette']:.3f}")
print(f"ARI: {scores['ari']:.3f}")
'''

    print(code)


def comparative_evaluation():
    """Show how to compare multiple models."""
    print("\n" + "=" * 70)
    print("=== COMPARATIVE EVALUATION ===\n")

    print("Compare multiple embedding models on same tests:\n")

    code = '''from sentence_transformers import SentenceTransformer

models = {
    "MiniLM": SentenceTransformer('all-MiniLM-L6-v2'),
    "MPNet": SentenceTransformer('all-mpnet-base-v2'),
    "Custom": SentenceTransformer('./my-finetuned-model'),
}

# Prepare test data
test_queries = [...]
test_docs = [...]
relevance_map = {...}

# Evaluate each model
results = {}
for name, model in models.items():
    mrr = calculate_mrr(model, test_queries, test_docs, relevance_map)
    recall_10 = calculate_recall_at_k(model, test_queries, test_docs, relevance_map, k=10)

    results[name] = {
        "MRR": mrr,
        "Recall@10": recall_10
    }

# Print comparison table
print("Model      MRR     Recall@10")
print("-" * 35)
for name, scores in results.items():
    print(f"{name:10} {scores['MRR']:.3f}   {scores['Recall@10']:.3f}")

# Output example:
# Model      MRR     Recall@10
# -----------------------------------
# MiniLM     0.652   0.723
# MPNet      0.701   0.768
# Custom     0.745   0.812  ← Best
'''

    print(code)


def intrinsic_vs_extrinsic():
    """Explain intrinsic vs extrinsic evaluation."""
    print("\n" + "=" * 70)
    print("=== INTRINSIC VS EXTRINSIC EVALUATION ===\n")

    print("Intrinsic Evaluation:")
    print("  Measure embedding quality directly")
    print("  Examples:")
    print("    • Semantic similarity correlation")
    print("    • Clustering metrics")
    print("    • Analogy tasks (king - man + woman = queen)")
    print("  ✓ Fast, standardized benchmarks")
    print("  ✗ May not reflect downstream performance\n")

    print("Extrinsic Evaluation:")
    print("  Measure performance on downstream task")
    print("  Examples:")
    print("    • Classification accuracy with embeddings as features")
    print("    • Search/retrieval quality in production")
    print("    • User satisfaction metrics")
    print("  ✓ Reflects real-world performance")
    print("  ✗ Slower, requires task-specific setup\n")

    print("Best practice:")
    print("  Use both! Intrinsic for quick comparison, extrinsic for final validation")


def evaluation_checklist():
    """Provide evaluation checklist."""
    print("\n" + "=" * 70)
    print("=== EVALUATION CHECKLIST ===\n")

    checklist = [
        "□ Test on representative data from your domain",
        "□ Use multiple metrics (similarity, retrieval, clustering)",
        "□ Compare to baseline (base model or previous version)",
        "□ Include edge cases and difficult examples",
        "□ Test with different query types",
        "□ Measure on held-out test set (not training data)",
        "□ Check for biases (gender, race, etc.)",
        "□ Validate with human judgment on sample",
        "□ Measure inference speed and memory",
        "□ Test at expected production scale",
        "□ Document all results and configurations",
        "□ A/B test in production if possible",
    ]

    for item in checklist:
        print(f"  {item}")


def real_world_example():
    """Show real-world evaluation example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE ===\n")

    print("Task: Evaluate embeddings for internal docs search\n")

    print("Test setup:")
    print("  • 500 documents (company policies, procedures)")
    print("  • 50 test queries from actual user searches")
    print("  • Human-annotated relevant documents per query\n")

    print("Models tested:")
    print("  1. OpenAI text-embedding-3-small (baseline)")
    print("  2. all-mpnet-base-v2 (open source)")
    print("  3. Custom fine-tuned model (domain adapted)\n")

    print("Results:\n")

    print("Model                  MRR     Recall@5   Recall@10   Latency")
    print("-" * 70)
    print("OpenAI                 0.712   0.654      0.782       45ms")
    print("MPNet base             0.685   0.623      0.751       8ms")
    print("Custom adapted         0.748   0.698      0.823       9ms")
    print("-" * 70)

    print("\n\nFindings:")
    print("  ✓ Custom model best quality (+5% vs OpenAI)")
    print("  ✓ 5× faster than OpenAI (local inference)")
    print("  ✓ No API costs")
    print("  ✓ Better handling of company-specific terms\n")

    print("Decision: Deploy custom adapted model")


def best_practices():
    """List evaluation best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Use domain-specific tests", "Generic benchmarks may not reflect your task"),
        ("Multiple metrics", "No single metric tells full story"),
        ("Baseline comparison", "Always compare to existing solution"),
        ("Human validation", "Verify metrics match human judgment"),
        ("Edge cases", "Test worst-case and ambiguous examples"),
        ("Reproducible", "Fix random seeds, document exact setup"),
        ("Statistical significance", "Use multiple runs, confidence intervals"),
        ("Monitor in production", "Evaluation doesn't stop at deployment"),
    ]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


def common_pitfalls():
    """Show common evaluation pitfalls."""
    print("=" * 70)
    print("=== COMMON PITFALLS ===\n")

    pitfalls = [
        ("Testing on training data",
         "Inflated metrics, poor generalization"),

        ("Single metric focus",
         "Miss important quality aspects"),

        ("Small test set",
         "Unreliable results, high variance"),

        ("Ignoring speed",
         "Great quality but too slow for production"),

        ("Generic benchmarks only",
         "May not correlate with your task performance"),

        ("No human validation",
         "Metrics don't always match user satisfaction"),

        ("Cherry-picking results",
         "Report only best metrics, mislead stakeholders"),
    ]

    for pitfall, consequence in pitfalls:
        print(f"✗ {pitfall}")
        print(f"  → {consequence}\n")


def practical_workflow():
    """Show practical evaluation workflow."""
    print("=" * 70)
    print("=== PRACTICAL WORKFLOW ===\n")

    steps = [
        "1. Define evaluation criteria (what matters for your task?)",
        "2. Collect/create test set (50-500 examples)",
        "3. Get human annotations if needed (relevance judgments)",
        "4. Choose metrics (MRR, Recall@K, etc.)",
        "5. Evaluate baseline model",
        "6. Evaluate your model(s)",
        "7. Compare results statistically",
        "8. Review failure cases manually",
        "9. Validate with human judges on sample",
        "10. Document findings and recommendations",
        "11. A/B test in production (if possible)",
        "12. Monitor metrics over time",
    ]

    for step in steps:
        print(f"  {step}")


if __name__ == "__main__":
    embedding_evaluation_intro()
    semantic_similarity_evaluation()
    retrieval_evaluation()
    clustering_evaluation()
    comparative_evaluation()
    intrinsic_vs_extrinsic()
    evaluation_checklist()
    real_world_example()
    best_practices()
    common_pitfalls()
    practical_workflow()

    print("\n" + "=" * 70)
    print("\nKey insight: Rigorous evaluation is critical!")
    print("Test on YOUR data, use multiple metrics, and validate with humans")
