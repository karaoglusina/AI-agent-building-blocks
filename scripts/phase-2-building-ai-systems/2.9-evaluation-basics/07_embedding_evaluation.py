"""
07 - Embedding Evaluation
=========================
Evaluate embedding quality for your domain.

Key concept: Generic embeddings may not capture domain-specific similarity.

Book reference: speach_lang.I.6.12, hands_on_LLM.III.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_DEPENDENCIES.append('sentence_transformers')


# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_triplets(model, triplets: list[tuple[str, str, str]]) -> dict:
    """
    Evaluate on triplets: (anchor, positive, negative).
    Positive should be more similar to anchor than negative.
    """
    correct = 0
    margins = []
    
    for anchor, positive, negative in triplets:
        embeddings = model.encode([anchor, positive, negative])
        
        sim_pos = cosine_similarity(embeddings[0], embeddings[1])
        sim_neg = cosine_similarity(embeddings[0], embeddings[2])
        
        if sim_pos > sim_neg:
            correct += 1
        margins.append(sim_pos - sim_neg)
    
    return {
        "accuracy": correct / len(triplets),
        "avg_margin": np.mean(margins),
        "min_margin": np.min(margins),
    }


def evaluate_similarity_pairs(model, pairs: list[tuple[str, str, float]]) -> dict:
    """
    Evaluate on similarity pairs: (text1, text2, expected_similarity).
    Measures correlation between predicted and expected similarity.
    """
    predicted = []
    expected = []
    
    for text1, text2, exp_sim in pairs:
        embeddings = model.encode([text1, text2])
        pred_sim = cosine_similarity(embeddings[0], embeddings[1])
        predicted.append(pred_sim)
        expected.append(exp_sim)
    
    correlation = np.corrcoef(predicted, expected)[0, 1]
    
    return {
        "correlation": correlation,
        "predicted": predicted,
        "expected": expected,
    }


# Domain-specific test data for job search
JOB_TRIPLETS = [
    # (anchor, positive=similar, negative=different)
    ("Python backend developer", "Django web developer", "Graphic designer"),
    ("Data scientist machine learning", "ML engineer deep learning", "HR manager"),
    ("Senior software engineer", "Staff engineer", "Marketing intern"),
    ("DevOps Kubernetes", "Cloud infrastructure engineer", "Sales representative"),
    ("Product manager", "Technical product owner", "Data entry clerk")]

SIMILARITY_PAIRS = [
    # (text1, text2, expected_similarity 0-1)
    ("Python developer", "Python software engineer", 0.9),
    ("Python developer", "Java developer", 0.6),
    ("Python developer", "Chef", 0.1),
    ("Remote work", "Work from home", 0.95),
    ("Senior position", "Entry level", 0.3)]


if __name__ == "__main__":
    print("=== EMBEDDING EVALUATION ===\n")
    
    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Triplet evaluation
    print("\n--- Triplet Evaluation ---")
    print("(Does positive rank higher than negative?)\n")
    
    triplet_results = evaluate_triplets(model, JOB_TRIPLETS)
    print(f"Accuracy: {triplet_results['accuracy']:.0%}")
    print(f"Average margin: {triplet_results['avg_margin']:.3f}")
    print(f"Minimum margin: {triplet_results['min_margin']:.3f}")
    
    # Show individual triplets
    print("\nTriplet details:")
    for anchor, pos, neg in JOB_TRIPLETS:
        emb = model.encode([anchor, pos, neg])
        sim_pos = cosine_similarity(emb[0], emb[1])
        sim_neg = cosine_similarity(emb[0], emb[2])
        status = "✓" if sim_pos > sim_neg else "✗"
        print(f"  {status} \"{anchor[:20]}...\" → pos:{sim_pos:.2f}, neg:{sim_neg:.2f}")
    
    # Similarity correlation
    print("\n--- Similarity Correlation ---")
    print("(How well do predicted similarities match expected?)\n")
    
    sim_results = evaluate_similarity_pairs(model, SIMILARITY_PAIRS)
    print(f"Correlation: {sim_results['correlation']:.3f}")
    
    print("\nPair details:")
    for (t1, t2, exp), pred in zip(SIMILARITY_PAIRS, sim_results['predicted']):
        diff = abs(pred - exp)
        status = "✓" if diff < 0.2 else "~" if diff < 0.4 else "✗"
        print(f"  {status} \"{t1}\" ↔ \"{t2}\": expected={exp:.2f}, got={pred:.2f}")
    
    print("\n=== EVALUATION TIPS ===")
    print("• Create domain-specific test sets")
    print("• High triplet accuracy = good ranking")
    print("• High correlation = good similarity estimation")
    print("• Consider fine-tuning if scores are low")
