"""
03 - Image-Text Search
=======================
Build CLIP-based multimodal search system.

Key concept: CLIP embeddings enable bidirectional search - find images with text queries, or find text descriptions from images.

Book reference: hands_on_LLM.II.9
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def multimodal_search_intro():
    """Introduce multimodal search."""
    print("=== MULTIMODAL SEARCH ===\n")

    print("What is multimodal search?")
    print("  Search across different modalities (text, images) in unified space.\n")

    print("Search capabilities:")
    print("  1. Text → Image: 'sunset over ocean' → [beach photos]")
    print("  2. Image → Image: [cat photo] → [similar cat photos]")
    print("  3. Image → Text: [product photo] → ['blue running shoes']\n")

    print("Why multimodal search?")
    print("  ✓ More natural search (describe what you want)")
    print("  ✓ Find similar items across modalities")
    print("  ✓ Better than keyword matching")
    print("  ✓ No manual tagging required")
    print("  ✓ Semantic understanding")


def building_search_index():
    """Show how to build search index."""
    print("\n" + "=" * 70)
    print("=== BUILDING SEARCH INDEX ===\n")

    code = '''import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Dict

class MultimodalSearchIndex:
    """CLIP-based search index for images."""

    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)

        self.image_embeddings = []
        self.image_metadata = []

    def add_images(self, image_paths: List[str], metadata: List[Dict] = None):
        """Add images to index."""
        print(f"Indexing {len(image_paths)} images...")

        for i, path in enumerate(image_paths):
            # Load and encode
            image = Image.open(path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)

            self.image_embeddings.append(features.cpu().numpy())

            # Store metadata
            meta = metadata[i] if metadata else {}
            meta['path'] = path
            self.image_metadata.append(meta)

        print(f"Indexed {len(self.image_embeddings)} images")

    def search_by_text(self, query: str, top_k: int = 10):
        """Search images using text query."""
        # Encode query
        text_input = clip.tokenize([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        text_emb = text_features.cpu().numpy()
        image_embs = np.vstack(self.image_embeddings)

        similarities = (image_embs @ text_emb.T).squeeze()

        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.image_metadata[idx],
                'score': float(similarities[idx])
            })

        return results

    def search_by_image(self, image_path: str, top_k: int = 10):
        """Search similar images using image query."""
        # Encode query image
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_features = self.model.encode_image(image_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)

        # Find similar
        query_emb = query_features.cpu().numpy()
        image_embs = np.vstack(self.image_embeddings)

        similarities = (image_embs @ query_emb.T).squeeze()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.image_metadata[idx],
                'score': float(similarities[idx])
            })

        return results

    def save(self, path: str):
        """Save index to disk."""
        np.savez(
            path,
            embeddings=np.vstack(self.image_embeddings),
            metadata=self.image_metadata
        )

    def load(self, path: str):
        """Load index from disk."""
        data = np.load(path, allow_pickle=True)
        self.image_embeddings = list(data['embeddings'])
        self.image_metadata = list(data['metadata'])
'''

    print(code)


def usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 70)
    print("=== USAGE EXAMPLES ===\n")

    code = '''# 1. Build index
index = MultimodalSearchIndex()

image_paths = [
    "photos/beach1.jpg",
    "photos/mountain1.jpg",
    "photos/city1.jpg",
    # ... thousands more
]

metadata = [
    {"id": 1, "category": "beach"},
    {"id": 2, "category": "mountain"},
    {"id": 3, "category": "city"}]

index.add_images(image_paths, metadata)

# Save for later
index.save("image_index.npz")

# 2. Text search
results = index.search_by_text("sunset over the ocean", top_k=5)

print("Text search results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['path']} (score: {result['score']:.3f})")

# 3. Image search
results = index.search_by_image("query_image.jpg", top_k=5)

print("\\nSimilar images:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['path']} (score: {result['score']:.3f})")
'''

    print(code)


def advanced_features():
    """Show advanced features."""
    print("\n" + "=" * 70)
    print("=== ADVANCED FEATURES ===\n")

    print("1. Hybrid search (text + filters)\n")

    code1 = '''def hybrid_search(index, query, filters=None, top_k=10):
    """Search with text query and metadata filters."""

    # Get semantic results
    results = index.search_by_text(query, top_k=top_k * 10)

    # Apply filters
    if filters:
        results = [
            r for r in results
            if all(r.get(k) == v for k, v in filters.items())
        ]

    return results[:top_k]

# Example
results = hybrid_search(
    index,
    query="red sports car",
    filters={"year": 2023, "brand": "Ferrari"},
    top_k=5
)
'''
    print(code1)

    print("\n2. Multi-query search\n")

    code2 = '''def multi_query_search(index, queries, top_k=10):
    """Search with multiple queries, combine results."""

    all_scores = {}

    for query in queries:
        results = index.search_by_text(query, top_k=top_k * 2)

        for result in results:
            path = result['path']
            score = result['score']

            if path in all_scores:
                all_scores[path] += score
            else:
                all_scores[path] = score

    # Sort by combined score
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    return [{"path": path, "score": score} for path, score in ranked[:top_k]]

# Example: Find images matching any query
results = multi_query_search(
    index,
    queries=["beach sunset", "ocean waves", "tropical paradise"],
    top_k=10
)
'''
    print(code2)

    print("\n3. Negative search\n")

    code3 = '''def search_with_negatives(index, positive_query, negative_queries, top_k=10):
    """Search for positive query but penalize negative queries."""

    # Encode positive
    pos_emb = encode_text(positive_query)

    # Encode negatives
    neg_embs = [encode_text(q) for q in negative_queries]

    # Compute scores
    scores = image_embeddings @ pos_emb

    # Penalize negatives
    for neg_emb in neg_embs:
        neg_scores = image_embeddings @ neg_emb
        scores -= 0.5 * neg_scores  # Weight can be tuned

    top_indices = scores.argsort()[::-1][:top_k]
    return [index.image_metadata[i] for i in top_indices]

# Example: Dogs but not puppies
results = search_with_negatives(
    index,
    positive_query="dog",
    negative_queries=["puppy", "small dog"],
    top_k=10
)
'''
    print(code3)


def vector_database_integration():
    """Show vector database integration."""
    print("\n" + "=" * 70)
    print("=== VECTOR DATABASE INTEGRATION ===\n")

    print("For production-scale search, use vector databases:\n")

    code = '''# Using FAISS (Facebook AI Similarity Search)
import faiss

class FAISSSearchIndex:
    """Fast approximate nearest neighbor search."""

    def __init__(self, dimension=512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
        self.metadata = []

    def add_embeddings(self, embeddings, metadata):
        """Add embeddings to FAISS index."""
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=10):
        """Search similar vectors."""
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                **self.metadata[idx],
                'score': float(score)
            })

        return results

# Benefits of FAISS:
# - 10-100× faster than brute force
# - Scales to millions/billions of vectors
# - Approximate search (configurable accuracy/speed)
# - GPU support for even faster search
'''

    print(code)

    print("\n\nOther vector databases:")
    print("  • Pinecone: Managed vector DB")
    print("  • Weaviate: Open source vector search")
    print("  • Milvus: Scalable vector DB")
    print("  • Qdrant: Rust-based vector engine")
    print("  • ChromaDB: Simple vector DB")


def evaluation_metrics():
    """Show evaluation metrics."""
    print("\n" + "=" * 70)
    print("=== EVALUATION METRICS ===\n")

    code = '''def evaluate_search(index, test_queries, ground_truth):
    """
    Evaluate search quality.

    test_queries: List of query strings
    ground_truth: Dict mapping queries to relevant image IDs
    """

    mrr_sum = 0  # Mean Reciprocal Rank
    recall_at_10_sum = 0

    for query in test_queries:
        results = index.search_by_text(query, top_k=10)
        result_ids = [r['id'] for r in results]

        relevant_ids = ground_truth[query]

        # MRR: Position of first relevant result
        for rank, doc_id in enumerate(result_ids, 1):
            if doc_id in relevant_ids:
                mrr_sum += 1.0 / rank
                break

        # Recall@10: Proportion of relevant docs in top 10
        found = len(set(result_ids) & set(relevant_ids))
        recall_at_10_sum += found / len(relevant_ids)

    mrr = mrr_sum / len(test_queries)
    recall_at_10 = recall_at_10_sum / len(test_queries)

    return {
        "MRR": mrr,
        "Recall@10": recall_at_10
    }

# Example
metrics = evaluate_search(index, test_queries, ground_truth)
print(f"MRR: {metrics['MRR']:.3f}")
print(f"Recall@10: {metrics['Recall@10']:.3f}")
'''

    print(code)


def real_world_example():
    """Show real-world example."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD EXAMPLE: E-COMMERCE SEARCH ===\n")

    print("Requirements:")
    print("  • 100K product images")
    print("  • Text search ('blue running shoes')")
    print("  • Image search (find similar products)")
    print("  • Filters (price, brand, category)")
    print("  • Fast (<100ms response time)\n")

    print("Implementation:")
    print("  1. Index all product images with CLIP")
    print("  2. Store embeddings in FAISS for speed")
    print("  3. Combine semantic search with filters")
    print("  4. Cache popular queries\n")

    print("Results:")
    print("  • Search latency: 45ms average")
    print("  • Relevance: +35% vs keyword search")
    print("  • User satisfaction: +28%")
    print("  • Conversion rate: +12%")


def best_practices():
    """List best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        ("Normalize embeddings", "Always normalize for cosine similarity"),
        ("Use FAISS for scale", "Essential for >10K images"),
        ("Cache embeddings", "Don't recompute on every search"),
        ("Hybrid search", "Combine semantic + metadata filters"),
        ("Monitor quality", "Track MRR, Recall@K"),
        ("Update index incrementally", "Don't rebuild entire index"),
        ("GPU for indexing", "10-100× faster embedding generation")]

    for practice, explanation in practices:
        print(f"✓ {practice}")
        print(f"  → {explanation}\n")


if __name__ == "__main__":
    multimodal_search_intro()
    building_search_index()
    usage_examples()
    advanced_features()
    vector_database_integration()
    evaluation_metrics()
    real_world_example()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: CLIP enables powerful multimodal search!")
    print("Build semantic image search with just text descriptions")
