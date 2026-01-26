"""
02 - Retrieval Metrics
======================
MRR, MAP, NDCG, and recall@k.

Key concept: Retrieval metrics measure how well you rank relevant items.

Book reference: AI_eng.3, hands_on_LLM.II.8, speach_lang.II.14.4
"""

import numpy as np


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for item in top_k if item in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: fraction of relevant items in top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/position of first relevant item."""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision: average of precision@k for each relevant item."""
    if not relevant:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            relevant_count += 1
            precisions.append(relevant_count / i)
    
    return sum(precisions) / len(relevant) if precisions else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@K: measures ranking quality with position discount."""
    def dcg(items, rel_set, k):
        score = 0.0
        for i, item in enumerate(items[:k], 1):
            if item in rel_set:
                score += 1.0 / np.log2(i + 1)
        return score
    
    actual = dcg(retrieved, relevant, k)
    
    # Ideal: all relevant items first
    ideal_list = list(relevant)[:k] + [i for i in retrieved if i not in relevant]
    ideal = dcg(ideal_list, relevant, k)
    
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(queries: list[dict]) -> dict:
    """Evaluate retrieval across multiple queries."""
    metrics = {"mrr": [], "map": [], "p@5": [], "r@5": [], "ndcg@5": []}
    
    for query in queries:
        retrieved = query["retrieved"]
        relevant = set(query["relevant"])
        
        metrics["mrr"].append(mrr(retrieved, relevant))
        metrics["map"].append(average_precision(retrieved, relevant))
        metrics["p@5"].append(precision_at_k(retrieved, relevant, 5))
        metrics["r@5"].append(recall_at_k(retrieved, relevant, 5))
        metrics["ndcg@5"].append(ndcg_at_k(retrieved, relevant, 5))
    
    return {k: np.mean(v) for k, v in metrics.items()}


# Test data: search results for job queries
TEST_QUERIES = [
    {
        "query": "Python developer Amsterdam",
        "retrieved": ["job1", "job5", "job2", "job8", "job3", "job10", "job4"],
        "relevant": {"job1", "job2", "job3", "job4"},
    },
    {
        "query": "Data scientist remote",
        "retrieved": ["job20", "job21", "job15", "job22", "job23", "job24", "job25"],
        "relevant": {"job15", "job22", "job24"},
    },
    {
        "query": "Senior ML engineer",
        "retrieved": ["job30", "job31", "job32", "job33", "job34", "job35"],
        "relevant": {"job30", "job31", "job32", "job33", "job34"},  # All relevant
    },
]


if __name__ == "__main__":
    print("=== RETRIEVAL METRICS ===\n")
    
    # Single query example
    query = TEST_QUERIES[0]
    retrieved = query["retrieved"]
    relevant = set(query["relevant"])
    
    print(f"Query: {query['query']}")
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}\n")
    
    print("--- Single Query Metrics ---")
    for k in [1, 3, 5]:
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        n = ndcg_at_k(retrieved, relevant, k)
        print(f"@{k}: P={p:.2f}, R={r:.2f}, NDCG={n:.2f}")
    
    print(f"\nMRR: {mrr(retrieved, relevant):.2f}")
    print(f"MAP: {average_precision(retrieved, relevant):.2f}")
    
    # Aggregate metrics
    print("\n--- Aggregate Metrics (3 queries) ---")
    results = evaluate_retrieval(TEST_QUERIES)
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.3f}")
    
    print("\n=== METRIC INTERPRETATION ===")
    print("MRR: Higher = first relevant result appears earlier")
    print("MAP: Higher = relevant results concentrated at top")
    print("NDCG: Higher = better overall ranking quality")
    print("P@K: Higher = fewer irrelevant in top K")
    print("R@K: Higher = more relevant found in top K")
