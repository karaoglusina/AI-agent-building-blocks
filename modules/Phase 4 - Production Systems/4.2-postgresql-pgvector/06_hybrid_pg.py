"""
06 - Hybrid Search PostgreSQL
==============================
Combine full-text search with vector similarity for better results.

Key concept: Hybrid search combines keyword matching (BM25/full-text) with
semantic similarity (vectors) for more robust retrieval.

Book reference: AI_eng.6
"""

import os
from typing import Optional
from sqlalchemy import create_engine, select, func, or_, text
from sqlalchemy.orm import Session
from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from modules.phase4.__init__ import DATABASE_URL, VectorDocument


def enable_fulltext_search() -> bool:
    """Enable full-text search on content column."""
    engine = create_engine(DATABASE_URL, echo=True)

    try:
        with engine.connect() as conn:
            # Add tsvector column
            conn.execute(text("""
                ALTER TABLE vector_documents
                ADD COLUMN IF NOT EXISTS content_tsv tsvector
            """))

            # Create trigger to auto-update tsvector
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION vector_documents_tsv_update() RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """))

            conn.execute(text("""
                DROP TRIGGER IF EXISTS vector_documents_tsv_trigger ON vector_documents
            """))

            conn.execute(text("""
                CREATE TRIGGER vector_documents_tsv_trigger
                BEFORE INSERT OR UPDATE ON vector_documents
                FOR EACH ROW EXECUTE FUNCTION vector_documents_tsv_update()
            """))

            # Create GIN index for fast full-text search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS vector_documents_tsv_idx
                ON vector_documents USING GIN(content_tsv)
            """))

            # Update existing rows
            conn.execute(text("""
                UPDATE vector_documents
                SET content_tsv = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''))
            """))

            conn.commit()
            print("\n✓ Full-text search enabled")
            return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def keyword_search(query: str, limit: int = 5) -> list[tuple[VectorDocument, float]]:
    """Search using PostgreSQL full-text search."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        # Use ts_rank for relevance scoring
        results = session.execute(text("""
            SELECT *, ts_rank(content_tsv, to_tsquery('english', :query)) as rank
            FROM vector_documents
            WHERE content_tsv @@ to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """), {"query": query.replace(" ", " & "), "limit": limit}).fetchall()

        docs = []
        for row in results:
            doc = VectorDocument(
                id=row[0],
                title=row[1],
                content=row[2],
                embedding=row[3]
            )
            docs.append((doc, float(row[5])))  # rank

        return docs


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate embedding for text."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def vector_search(query: str, limit: int = 5) -> list[tuple[VectorDocument, float]]:
    """Search using vector similarity."""
    engine = create_engine(DATABASE_URL, echo=False)
    query_embedding = get_embedding(query)

    with Session(engine) as session:
        results = session.execute(
            select(
                VectorDocument,
                VectorDocument.embedding.cosine_distance(query_embedding).label("distance")
            )
            .order_by("distance")
            .limit(limit)
        ).all()

        return [(doc, float(dist)) for doc, dist in results]


def hybrid_search(
    query: str,
    limit: int = 5,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5
) -> list[tuple[VectorDocument, float]]:
    """
    Combine vector and keyword search with weighted scoring.

    Args:
        query: Search query
        limit: Maximum results
        vector_weight: Weight for vector similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
    """
    print(f"\nHybrid search: '{query}'")
    print(f"  Vector weight: {vector_weight}")
    print(f"  Keyword weight: {keyword_weight}")

    # Get vector results
    vector_results = vector_search(query, limit=limit * 2)
    vector_scores = {doc.id: (1 - dist) for doc, dist in vector_results}

    # Get keyword results
    keyword_results = keyword_search(query, limit=limit * 2)
    keyword_scores = {doc.id: rank for doc, rank in keyword_results}

    # Normalize scores
    max_vector = max(vector_scores.values()) if vector_scores else 1
    max_keyword = max(keyword_scores.values()) if keyword_scores else 1

    normalized_vector = {k: v / max_vector for k, v in vector_scores.items()}
    normalized_keyword = {k: v / max_keyword for k, v in keyword_scores.items()}

    # Combine scores
    all_doc_ids = set(normalized_vector.keys()) | set(normalized_keyword.keys())
    combined_scores = {}

    for doc_id in all_doc_ids:
        vec_score = normalized_vector.get(doc_id, 0) * vector_weight
        key_score = normalized_keyword.get(doc_id, 0) * keyword_weight
        combined_scores[doc_id] = vec_score + key_score

    # Get top documents
    top_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    # Fetch documents
    engine = create_engine(DATABASE_URL, echo=False)
    with Session(engine) as session:
        results = []
        for doc_id, score in top_ids:
            doc = session.get(VectorDocument, doc_id)
            if doc:
                results.append((doc, score))

    return results


def display_results(
    results: list[tuple[VectorDocument, float]],
    score_label: str = "Score"
) -> None:
    """Display search results."""
    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. {doc.title}")
        print(f"   {score_label}: {score:.4f}")
        print(f"   Content: {doc.content[:80]}...")


if __name__ == "__main__":
    print("Hybrid Search PostgreSQL Demo")
    print("=" * 50)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ Error: OPENAI_API_KEY not set")
        exit(1)

    # Enable full-text search
    print("\n1. ENABLE FULL-TEXT SEARCH")
    if not enable_fulltext_search():
        print("Warning: Full-text setup may have issues")

    # Test queries
    query = "learning algorithms"

    print("\n2. KEYWORD SEARCH ONLY")
    results = keyword_search(query, limit=3)
    display_results(results, "Rank")

    print("\n3. VECTOR SEARCH ONLY")
    results = vector_search(query, limit=3)
    display_results(results, "Similarity")

    print("\n4. HYBRID SEARCH (50/50)")
    results = hybrid_search(query, limit=3, vector_weight=0.5, keyword_weight=0.5)
    display_results(results, "Combined Score")

    print("\n5. HYBRID SEARCH (70% Vector, 30% Keyword)")
    results = hybrid_search(query, limit=3, vector_weight=0.7, keyword_weight=0.3)
    display_results(results, "Combined Score")

    print("\n" + "=" * 50)
    print("✓ Demo complete!")
    print("\nHybrid search combines:")
    print("  - Keyword matching (exact terms, BM25)")
    print("  - Semantic similarity (meaning, context)")
    print("  - Weighted scoring (configurable)")
