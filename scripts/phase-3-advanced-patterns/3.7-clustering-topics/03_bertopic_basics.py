"""
03 - BERTopic Basics
=====================
Automatic topic discovery from documents.

Key concept: BERTopic finds coherent topics automatically using embeddings + clustering + topic extraction.

Book reference: hands_on_LLM.II.5, NLP_cook.6
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    from bertopic import BERTopic
except ImportError:
    MISSING_DEPENDENCIES.append('bertopic')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_DEPENDENCIES.append('sentence_transformers')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def discover_topics(jobs: list[dict], min_topic_size: int = 5):
    """Discover topics from job descriptions using BERTopic."""
    # Prepare documents
    docs = [
        f"{job['title']}. {job.get('description', '')[:500]}"
        for job in jobs
    ]

    print(f"Analyzing {len(docs)} job descriptions...")
    print(f"Min topic size: {min_topic_size}\n")

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=True
    )

    # Fit model and get topics
    topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics, docs


def analyze_topics(topic_model: BERTopic, topics: list[int], docs: list[str]):
    """Analyze discovered topics."""
    print("\n" + "=" * 70)
    print("=== DISCOVERED TOPICS ===\n")

    topic_info = topic_model.get_topic_info()

    # -1 is outlier topic
    actual_topics = topic_info[topic_info['Topic'] != -1]

    print(f"Found {len(actual_topics)} topics\n")

    # Show each topic with top words and sample documents
    for _, row in actual_topics.iterrows():
        topic_id = row['Topic']
        count = row['Count']

        print(f"TOPIC {topic_id} ({count} documents):")

        # Get top words for topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = [word for word, score in topic_words[:5]]
            print(f"  Top words: {', '.join(words)}")

        # Show sample documents
        topic_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id]
        print(f"  Sample documents:")
        for doc in topic_docs[:2]:
            print(f"    - {doc[:80]}...")

        print()

    # Outlier count
    outlier_count = len([t for t in topics if t == -1])
    if outlier_count > 0:
        print(f"Outliers (Topic -1): {outlier_count} documents")
        print("  (Documents that don't fit any topic)")


def get_representative_docs(topic_model: BERTopic, topic_id: int, n: int = 3):
    """Get most representative documents for a topic."""
    repr_docs = topic_model.get_representative_docs(topic_id)

    print(f"\n=== MOST REPRESENTATIVE DOCS FOR TOPIC {topic_id} ===\n")

    for i, doc in enumerate(repr_docs[:n], 1):
        print(f"{i}. {doc[:150]}...\n")


if __name__ == "__main__":
    print("=== BERTOPIC TOPIC DISCOVERY ===\n")

    # Load jobs
    jobs = load_sample_jobs(100)
    print(f"Loaded {len(jobs)} jobs\n")

    # Discover topics
    topic_model, topics, docs = discover_topics(jobs, min_topic_size=5)

    # Analyze topics
    analyze_topics(topic_model, topics, docs)

    # Show representative docs for topic 0
    if len(set(topics)) > 1:  # More than just outliers
        actual_topics = [t for t in set(topics) if t != -1]
        if actual_topics:
            print("\n" + "=" * 70)
            get_representative_docs(topic_model, actual_topics[0], n=3)

    print("\n" + "=" * 70)
    print("BERTopic automatically discovered meaningful job market segments!")
    print("\nKey insight: Topics emerge from data patterns - no manual labeling needed")
