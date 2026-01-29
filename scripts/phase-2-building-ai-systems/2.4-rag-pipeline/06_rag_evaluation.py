"""
06 - RAG Evaluation
===================
Measure retrieval quality with standard metrics.

Key concept: Evaluate retrieval separately from generation - each can fail independently.

Book reference: AI_eng.3, AI_eng.4, hands_on_LLM.II.8
"""

import numpy as np


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: What fraction of top-k results are relevant?"""
    retrieved_k = retrieved[:k]
    relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_in_k / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: What fraction of relevant docs are in top-k?"""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    found = len(retrieved_k & relevant)
    return found / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: How high is the first relevant result?"""
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    def dcg(docs, rel_set, k):
        score = 0.0
        for i, doc in enumerate(docs[:k], 1):
            if doc in rel_set:
                score += 1.0 / np.log2(i + 1)
        return score
    
    # Ideal ranking: all relevant docs first
    ideal_docs = list(relevant)[:k] + [d for d in retrieved if d not in relevant]
    
    actual_dcg = dcg(retrieved, relevant, k)
    ideal_dcg = dcg(ideal_docs, relevant, k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# Test cases: simulated retrieval results
TEST_CASES = [
    {
        "query": "Python developer jobs",
        "retrieved": ["job_1", "job_5", "job_3", "job_8", "job_2"],
        "relevant": {"job_1", "job_2", "job_3"},
    },
    {
        "query": "Data scientist Amsterdam",
        "retrieved": ["job_10", "job_11", "job_4", "job_12", "job_13"],
        "relevant": {"job_4", "job_5", "job_6"},
    },
    {
        "query": "Remote ML engineer",
        "retrieved": ["job_20", "job_21", "job_22", "job_23", "job_24"],
        "relevant": {"job_20", "job_21", "job_22", "job_23", "job_24"},
    }]


if __name__ == "__main__":
    print("=== RAG EVALUATION METRICS ===\n")
    
    for case in TEST_CASES:
        query = case["query"]
        retrieved = case["retrieved"]
        relevant = case["relevant"]
        
        print(f"Query: {query}")
        print(f"  Retrieved: {retrieved}")
        print(f"  Relevant:  {relevant}")
        print()
        
        for k in [1, 3, 5]:
            p_k = precision_at_k(retrieved, relevant, k)
            r_k = recall_at_k(retrieved, relevant, k)
            n_k = ndcg_at_k(retrieved, relevant, k)
            print(f"  @{k}: P={p_k:.2f}, R={r_k:.2f}, NDCG={n_k:.2f}")
        
        rr = mrr(retrieved, relevant)
        print(f"  MRR: {rr:.2f}")
        print("-" * 50)
    
    print("\n=== METRIC INTERPRETATION ===")
    print("Precision@K: Higher = fewer irrelevant results in top K")
    print("Recall@K: Higher = more relevant docs found in top K")
    print("MRR: Higher = first relevant result appears earlier")
    print("NDCG@K: Higher = relevant results ranked better overall")
