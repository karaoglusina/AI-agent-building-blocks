"""
01 - K-Means Clustering
=======================
Group similar documents using K-Means on embeddings.

Key concept: K-Means clusters documents by similarity in embedding space - discover natural groupings without labels.

Book reference: NLP_cook.4, hands_on_LLM.II.5
"""

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def cluster_jobs(jobs: list[dict], n_clusters: int = 5) -> dict:
    """Cluster jobs using K-Means on embeddings."""
    # Create text representation
    texts = [
        f"{job['title']}. {job.get('description', '')[:300]}"
        for job in jobs
    ]

    print(f"Generating embeddings for {len(texts)} jobs...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nClustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Organize by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append({
            "title": jobs[idx]["title"],
            "company": jobs[idx].get("company", "Unknown"),
            "description": jobs[idx].get("description", "")[:200]
        })

    # Calculate cluster statistics
    stats = {
        "n_clusters": n_clusters,
        "cluster_sizes": [len(clusters[i]) for i in range(n_clusters)],
        "total_jobs": len(jobs),
        "inertia": kmeans.inertia_  # Sum of squared distances to centers
    }

    return {"clusters": clusters, "stats": stats, "labels": labels}


def find_optimal_k(embeddings: np.ndarray, max_k: int = 10) -> list[dict]:
    """Find optimal number of clusters using elbow method."""
    print("Testing different values of k...")
    results = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        results.append({
            "k": k,
            "inertia": kmeans.inertia_
        })

        print(f"k={k}: inertia={kmeans.inertia_:.2f}")

    return results


if __name__ == "__main__":
    print("=== K-MEANS CLUSTERING ===\n")

    # Load jobs
    jobs = load_sample_jobs(100)
    print(f"Loaded {len(jobs)} jobs\n")

    # Cluster jobs
    result = cluster_jobs(jobs, n_clusters=5)

    print("\n" + "=" * 70)
    print("=== CLUSTERING RESULTS ===\n")

    print("Cluster sizes:")
    for i, size in enumerate(result["stats"]["cluster_sizes"]):
        print(f"  Cluster {i}: {size} jobs")

    print(f"\nInertia (lower = tighter clusters): {result['stats']['inertia']:.2f}")

    # Show sample from each cluster
    print("\n" + "=" * 70)
    print("=== CLUSTER SAMPLES ===\n")

    for cluster_id in range(result["stats"]["n_clusters"]):
        print(f"CLUSTER {cluster_id} ({len(result['clusters'][cluster_id])} jobs):")
        for job in result["clusters"][cluster_id][:3]:  # Show 3 samples
            print(f"  - {job['title']} @ {job['company']}")
        print()

    # Optional: Find optimal k
    print("=" * 70)
    print("=== FINDING OPTIMAL K (Elbow Method) ===\n")
    texts = [f"{job['title']}. {job.get('description', '')[:300]}" for job in jobs]
    embeddings = model.encode(texts, show_progress_bar=False)
    elbow_results = find_optimal_k(embeddings, max_k=8)

    print("\nLook for the 'elbow' where inertia drops significantly.")
    print("That's typically the optimal k!")
