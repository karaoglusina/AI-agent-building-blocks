"""
05 - Topic Coherence
=====================
Evaluate topic quality using coherence metrics.

Key concept: Not all topics are equally good - coherence measures how semantically related topic words are.

Book reference: NLP_cook.6
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_topic_coherence(topic_words: list[str], embedding_model) -> float:
    """
    Calculate coherence of a topic using word embeddings.

    Coherence = average pairwise similarity of top topic words.
    Higher = more semantically related = better topic.
    """
    # Get embeddings for topic words
    embeddings = embedding_model.encode(topic_words)

    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Get upper triangle (avoid diagonal and duplicates)
    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]

    # Average similarity is coherence
    coherence = upper_triangle.mean()

    return coherence


def evaluate_topics(jobs: list[dict], min_topic_size: int = 5):
    """Discover topics and evaluate their coherence."""
    docs = [
        f"{job['title']}. {job.get('description', '')[:500]}"
        for job in jobs
    ]

    print(f"Discovering topics from {len(docs)} jobs...\n")

    # Create and fit BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(docs)

    return topic_model, topics


def analyze_topic_quality(topic_model: BERTopic):
    """Analyze quality of discovered topics."""
    print("=== TOPIC QUALITY ANALYSIS ===\n")

    topic_info = topic_model.get_topic_info()
    actual_topics = topic_info[topic_info['Topic'] != -1]

    coherence_scores = []

    for _, row in actual_topics.iterrows():
        topic_id = row['Topic']
        count = row['Count']

        # Get top words
        topic_words_scores = topic_model.get_topic(topic_id)
        if not topic_words_scores:
            continue

        # Extract just the words (top 10)
        topic_words = [word for word, score in topic_words_scores[:10]]

        # Calculate coherence
        coherence = calculate_topic_coherence(topic_words, embedding_model)
        coherence_scores.append(coherence)

        # Display word scores and coherence
        print(f"TOPIC {topic_id} ({count} docs):")
        print(f"  Coherence: {coherence:.3f}")
        print(f"  Top words:")
        for word, score in topic_words_scores[:5]:
            print(f"    - {word}: {score:.3f}")
        print()

    # Overall statistics
    if coherence_scores:
        print("=" * 70)
        print("=== OVERALL TOPIC QUALITY ===\n")
        print(f"Number of topics: {len(coherence_scores)}")
        print(f"Mean coherence: {np.mean(coherence_scores):.3f}")
        print(f"Min coherence: {np.min(coherence_scores):.3f}")
        print(f"Max coherence: {np.max(coherence_scores):.3f}")
        print(f"Std coherence: {np.std(coherence_scores):.3f}")

        # Interpretation
        print("\n=== INTERPRETATION ===")
        mean_coh = np.mean(coherence_scores)
        if mean_coh > 0.5:
            print("✓ High coherence - topics are well-defined")
        elif mean_coh > 0.3:
            print("~ Moderate coherence - topics are okay but could be better")
        else:
            print("✗ Low coherence - topics are poorly defined")

        print("\nTips to improve coherence:")
        print("  - Increase min_topic_size")
        print("  - Clean text more aggressively")
        print("  - Use more documents")
        print("  - Try different embedding models")


if __name__ == "__main__":
    print("=== TOPIC COHERENCE EVALUATION ===\n")

    # Load jobs
    jobs = load_sample_jobs(100)
    print(f"Loaded {len(jobs)} jobs\n")

    # Discover topics
    topic_model, topics = evaluate_topics(jobs, min_topic_size=5)

    # Analyze quality
    analyze_topic_quality(topic_model)

    print("\n" + "=" * 70)
    print("Coherence measures topic quality - use it to tune your model!")
    print("\nKey insight: Higher coherence = more interpretable topics")
