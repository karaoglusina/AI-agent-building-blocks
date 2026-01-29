"""
02 - UMAP Visualization
========================
Reduce dimensions for plotting document clusters.

Key concept: UMAP reduces high-dimensional embeddings to 2D/3D for visualization while preserving local structure.

Book reference: hands_on_LLM.II.5, speach_lang.I.6.9
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import umap
except ImportError:
    MISSING_DEPENDENCIES.append('umap')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_DEPENDENCIES.append('sentence_transformers')

try:
    from sklearn.cluster import KMeans
except ImportError:
    MISSING_DEPENDENCIES.append('sklearn')

import matplotlib.pyplot as plt
import numpy as np
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')


def visualize_clusters(jobs: list[dict], n_clusters: int = 5, save_path: str = "job_clusters.png"):
    """Create 2D visualization of job clusters."""
    # Create embeddings
    texts = [
        f"{job['title']}. {job.get('description', '')[:300]}"
        for job in jobs
    ]

    print(f"Generating embeddings for {len(texts)} jobs...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Cluster
    print(f"\nClustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Reduce to 2D with UMAP
    print("\nReducing to 2D with UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))

    # Plot each cluster with different color
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=50
        )

    plt.title('Job Postings Clustered in 2D Space (UMAP)', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    return embedding_2d, labels


def analyze_cluster_separation(embedding_2d: np.ndarray, labels: np.ndarray):
    """Analyze how well-separated the clusters are."""
    n_clusters = len(np.unique(labels))

    # Calculate cluster centers in 2D
    centers = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        center = embedding_2d[mask].mean(axis=0)
        centers.append(center)

    centers = np.array(centers)

    # Calculate inter-cluster distances
    print("\n=== CLUSTER SEPARATION ===\n")
    print("Average distance between cluster centers:")

    distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
            print(f"  Cluster {i} ↔ Cluster {j}: {dist:.2f}")

    print(f"\nMean inter-cluster distance: {np.mean(distances):.2f}")
    print("Higher = better separated clusters")


if __name__ == "__main__":
    print("=== UMAP VISUALIZATION ===\n")

    # Load jobs
    jobs = load_sample_jobs(100)
    print(f"Loaded {len(jobs)} jobs\n")

    # Visualize
    embedding_2d, labels = visualize_clusters(jobs, n_clusters=5)

    # Analyze separation
    analyze_cluster_separation(embedding_2d, labels)

    print("\n" + "=" * 70)
    print("UMAP preserves local structure - similar jobs cluster together!")
    print("Open job_clusters.png to see the visualization.")
