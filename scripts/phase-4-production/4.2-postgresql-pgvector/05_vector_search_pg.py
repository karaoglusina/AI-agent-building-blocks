"""
05 - Vector Search in PostgreSQL
=================================
Semantic search with pgvector using cosine similarity.

Key concept: pgvector enables semantic search directly in PostgreSQL,
combining the power of vector embeddings with relational data.

Book reference: AI_eng.6
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import utils._load_env  # Loads .env file automatically
import os
from typing import Optional
try:
    from sqlalchemy import create_engine, select
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

try:
    from sqlalchemy.orm import Session
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

from openai import OpenAI
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from modules.phase4.__init__ import DATABASE_URL, VectorDocument


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate embedding for text using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def insert_document_with_embedding(
    title: str,
    content: str
) -> Optional[VectorDocument]:
    """Insert a document with its embedding."""
    engine = create_engine(DATABASE_URL, echo=False)

    try:
        # Generate embedding
        print(f"Generating embedding for: {title}")
        embedding = get_embedding(content)

        # Insert document
        with Session(engine) as session:
            doc = VectorDocument(
                title=title,
                content=content,
                embedding=embedding
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            print(f"✓ Inserted: {doc}")
            return doc

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def vector_search(
    query: str,
    limit: int = 5,
    distance_threshold: float = 0.5
) -> list[tuple[VectorDocument, float]]:
    """Search documents by vector similarity."""
    engine = create_engine(DATABASE_URL, echo=False)

    # Generate query embedding
    print(f"\nSearching for: '{query}'")
    query_embedding = get_embedding(query)

    with Session(engine) as session:
        # Find nearest neighbors using cosine distance
        # Lower distance = more similar
        results = session.execute(
            select(
                VectorDocument,
                VectorDocument.embedding.cosine_distance(query_embedding).label("distance")
            )
            .order_by("distance")
            .limit(limit)
        ).all()

        # Filter by threshold
        filtered = [
            (doc, float(dist))
            for doc, dist in results
            if float(dist) < distance_threshold
        ]

        print(f"\n✓ Found {len(filtered)} results (threshold: {distance_threshold})")
        return filtered


def display_results(results: list[tuple[VectorDocument, float]]) -> None:
    """Display search results."""
    if not results:
        print("No results found.")
        return

    for i, (doc, distance) in enumerate(results, 1):
        similarity = 1 - distance  # Convert distance to similarity
        print(f"\n{i}. {doc.title}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Content: {doc.content[:100]}...")


def insert_sample_documents() -> None:
    """Insert sample documents for testing."""
    samples = [
        ("Machine Learning Basics", "Machine learning algorithms learn from data to make predictions."),
        ("Deep Learning Neural Networks", "Deep learning uses neural networks with multiple layers."),
        ("Natural Language Processing", "NLP enables computers to understand human language."),
        ("Computer Vision", "Computer vision allows machines to interpret visual information."),
        ("Reinforcement Learning", "RL trains agents through rewards and penalties."),
        ("Python Programming", "Python is a versatile programming language for data science."),
        ("Database Management", "Databases store and organize structured data efficiently."),
        ("Web Development", "Web development involves creating websites and applications.")]

    print("Inserting sample documents...")
    for title, content in samples:
        insert_document_with_embedding(title, content)
    print(f"\n✓ Inserted {len(samples)} documents")


if __name__ == "__main__":
    print("Vector Search in PostgreSQL Demo")
    print("=" * 50)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        exit(1)

    # Insert sample data
    print("\n1. INSERTING SAMPLE DOCUMENTS")
    insert_sample_documents()

    # Search examples
    queries = [
        "artificial intelligence and neural networks",
        "programming languages",
        "storing data"
    ]

    print("\n2. VECTOR SEARCH EXAMPLES")
    for query in queries:
        print("\n" + "=" * 50)
        results = vector_search(query, limit=3, distance_threshold=0.6)
        display_results(results)

    print("\n" + "=" * 50)
    print("✓ Demo complete!")
