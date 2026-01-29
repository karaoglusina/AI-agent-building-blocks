"""
06 - Interactive Exploration
=============================
Explore clusters programmatically to find insights.

Key concept: Clusters are just the start - explore them interactively to find patterns, outliers, and insights.

Book reference: NLP_cook.7
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_DEPENDENCIES.append('sentence_transformers')

try:
    from sklearn.cluster import KMeans
except ImportError:
    MISSING_DEPENDENCIES.append('sklearn')

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    MISSING_DEPENDENCIES.append('sklearn')

import numpy as np
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

model = SentenceTransformer('all-MiniLM-L6-v2')


class ClusterExplorer:
    """Interactive cluster exploration tool."""

    def __init__(self, jobs: list[dict], n_clusters: int = 5):
        """Initialize explorer with jobs."""
        self.jobs = jobs
        self.n_clusters = n_clusters

        # Create embeddings
        print(f"Encoding {len(jobs)} jobs...")
        self.texts = [
            f"{job['title']}. {job.get('description', '')[:300]}"
            for job in jobs
        ]
        self.embeddings = model.encode(self.texts, show_progress_bar=False)

        # Cluster
        print(f"Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        print("Ready to explore!\n")

    def get_cluster(self, cluster_id: int) -> list[dict]:
        """Get all jobs in a cluster."""
        indices = [i for i, label in enumerate(self.labels) if label == cluster_id]
        return [self.jobs[i] for i in indices]

    def find_central_jobs(self, cluster_id: int, n: int = 5) -> list[dict]:
        """Find most central (representative) jobs in a cluster."""
        cluster_center = self.cluster_centers[cluster_id].reshape(1, -1)

        # Get jobs in cluster
        indices = [i for i, label in enumerate(self.labels) if label == cluster_id]

        # Calculate distances to center
        cluster_embeddings = self.embeddings[indices]
        distances = cosine_similarity(cluster_embeddings, cluster_center).flatten()

        # Get top N closest
        top_indices = np.argsort(-distances)[:n]  # Negative for descending
        central_jobs = [self.jobs[indices[i]] for i in top_indices]

        return central_jobs

    def find_outliers(self, cluster_id: int, n: int = 3) -> list[dict]:
        """Find outlier jobs in a cluster (far from center)."""
        cluster_center = self.cluster_centers[cluster_id].reshape(1, -1)

        indices = [i for i, label in enumerate(self.labels) if label == cluster_id]
        cluster_embeddings = self.embeddings[indices]
        distances = cosine_similarity(cluster_embeddings, cluster_center).flatten()

        # Get bottom N (farthest from center)
        outlier_indices = np.argsort(distances)[:n]
        outliers = [self.jobs[indices[i]] for i in outlier_indices]

        return outliers

    def compare_clusters(self, cluster_a: int, cluster_b: int):
        """Compare two clusters."""
        # Get similarity between cluster centers
        center_a = self.cluster_centers[cluster_a].reshape(1, -1)
        center_b = self.cluster_centers[cluster_b].reshape(1, -1)
        similarity = cosine_similarity(center_a, center_b)[0][0]

        # Get cluster sizes
        size_a = sum(1 for label in self.labels if label == cluster_a)
        size_b = sum(1 for label in self.labels if label == cluster_b)

        print(f"\n=== CLUSTER COMPARISON ===")
        print(f"Cluster {cluster_a} size: {size_a}")
        print(f"Cluster {cluster_b} size: {size_b}")
        print(f"Center similarity: {similarity:.3f}")

        if similarity > 0.8:
            print("→ Very similar clusters - consider merging")
        elif similarity > 0.6:
            print("→ Somewhat similar - some overlap expected")
        else:
            print("→ Distinct clusters")

    def search_similar_to_job(self, job_title: str, n: int = 5) -> list[dict]:
        """Find jobs similar to a given job title."""
        # Find the job
        job_idx = None
        for i, job in enumerate(self.jobs):
            if job_title.lower() in job['title'].lower():
                job_idx = i
                break

        if job_idx is None:
            print(f"Job containing '{job_title}' not found")
            return []

        # Find similar jobs
        query_embedding = self.embeddings[job_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # Get top N (excluding the query job itself)
        top_indices = np.argsort(-similarities)[1:n+1]
        similar_jobs = [self.jobs[i] for i in top_indices]

        return similar_jobs


if __name__ == "__main__":
    print("=== INTERACTIVE CLUSTER EXPLORATION ===\n")

    # Load jobs and create explorer
    jobs = load_sample_jobs(100)
    explorer = ClusterExplorer(jobs, n_clusters=5)

    # 1. Find central jobs in cluster 0
    print("=" * 70)
    print("1. MOST REPRESENTATIVE JOBS IN CLUSTER 0:\n")
    central = explorer.find_central_jobs(0, n=3)
    for job in central:
        print(f"  - {job['title']} @ {job.get('company', 'Unknown')}")

    # 2. Find outliers in cluster 0
    print("\n" + "=" * 70)
    print("2. OUTLIERS IN CLUSTER 0 (unusual for this cluster):\n")
    outliers = explorer.find_outliers(0, n=3)
    for job in outliers:
        print(f"  - {job['title']} @ {job.get('company', 'Unknown')}")

    # 3. Compare clusters
    print("\n" + "=" * 70)
    print("3. CLUSTER COMPARISON:")
    explorer.compare_clusters(0, 1)

    # 4. Find similar jobs
    print("\n" + "=" * 70)
    print("4. JOBS SIMILAR TO FIRST JOB:\n")
    first_job = jobs[0]
    print(f"Query job: {first_job['title']}\n")
    similar = explorer.search_similar_to_job(first_job['title'], n=5)
    for job in similar:
        print(f"  - {job['title']} @ {job.get('company', 'Unknown')}")

    print("\n" + "=" * 70)
    print("Interactive exploration reveals insights that summary stats miss!")
    print("\nKey insight: Explore clusters to understand your data deeply")
