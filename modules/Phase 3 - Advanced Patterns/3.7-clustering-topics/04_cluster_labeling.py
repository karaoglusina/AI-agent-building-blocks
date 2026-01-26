"""
04 - Cluster Labeling
======================
Use LLM to generate descriptive names for clusters.

Key concept: Clusters have numbers, but humans need names - LLMs can analyze cluster content and generate meaningful labels.

Book reference: hands_on_LLM.II.5
"""

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()
model = SentenceTransformer('all-MiniLM-L6-v2')


def cluster_jobs(jobs: list[dict], n_clusters: int = 5) -> dict:
    """Cluster jobs and organize by cluster."""
    texts = [
        f"{job['title']}. {job.get('description', '')[:300]}"
        for job in jobs
    ]

    embeddings = model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Organize by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(jobs[idx])

    return clusters


def generate_cluster_label(cluster_jobs: list[dict], n_samples: int = 10) -> dict:
    """Generate descriptive label for cluster using LLM."""
    # Sample jobs from cluster
    samples = cluster_jobs[:n_samples]

    # Create sample text
    sample_text = "\n".join([
        f"- {job['title']} @ {job.get('company', 'Unknown')}"
        for job in samples
    ])

    # Generate label
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a job market analyst. Given a list of job titles, identify the common theme and create a descriptive label.

Format your response as:
Label: [2-4 word descriptive name]
Description: [1 sentence explaining what these jobs have in common]
Key skills: [3-5 common skills]"""
            },
            {
                "role": "user",
                "content": f"Analyze these job titles and identify the cluster theme:\n\n{sample_text}"
            }
        ],
        temperature=0.5
    )

    content = response.choices[0].message.content

    # Parse response
    lines = content.split('\n')
    label_line = [l for l in lines if l.startswith('Label:')]
    desc_line = [l for l in lines if l.startswith('Description:')]
    skills_line = [l for l in lines if l.startswith('Key skills:')]

    return {
        "label": label_line[0].replace('Label:', '').strip() if label_line else "Unknown",
        "description": desc_line[0].replace('Description:', '').strip() if desc_line else "",
        "key_skills": skills_line[0].replace('Key skills:', '').strip() if skills_line else "",
        "size": len(cluster_jobs)
    }


def label_all_clusters(clusters: dict) -> dict:
    """Generate labels for all clusters."""
    labeled_clusters = {}

    for cluster_id, jobs in clusters.items():
        print(f"Labeling cluster {cluster_id} ({len(jobs)} jobs)...")

        cluster_info = generate_cluster_label(jobs)
        cluster_info["jobs"] = jobs

        labeled_clusters[cluster_id] = cluster_info

    return labeled_clusters


if __name__ == "__main__":
    print("=== CLUSTER LABELING WITH LLM ===\n")

    # Load and cluster jobs
    jobs = load_sample_jobs(100)
    print(f"Loaded {len(jobs)} jobs\n")

    print("Clustering jobs...")
    clusters = cluster_jobs(jobs, n_clusters=5)
    print(f"Created {len(clusters)} clusters\n")

    # Label clusters
    print("Generating labels...\n")
    labeled_clusters = label_all_clusters(clusters)

    # Display results
    print("\n" + "=" * 70)
    print("=== LABELED CLUSTERS ===\n")

    for cluster_id in sorted(labeled_clusters.keys()):
        info = labeled_clusters[cluster_id]

        print(f"CLUSTER {cluster_id}: {info['label']}")
        print(f"Size: {info['size']} jobs")
        print(f"Description: {info['description']}")
        print(f"Key skills: {info['key_skills']}")

        # Show sample jobs
        print("Sample jobs:")
        for job in info['jobs'][:3]:
            print(f"  - {job['title']} @ {job.get('company', 'Unknown')}")

        print()

    print("=" * 70)
    print("LLM-generated labels make clusters interpretable!")
    print("\nKey insight: Clusters + LLM = Automatic taxonomy generation")
