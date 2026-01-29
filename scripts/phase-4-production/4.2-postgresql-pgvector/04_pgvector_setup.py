"""
04 - pgvector Setup
===================
Enable pgvector extension and create vector columns.

Key concept: pgvector adds vector similarity search capabilities to PostgreSQL,
enabling efficient semantic search directly in your database.

Book reference: AI_eng.6
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

from typing import Optional
try:
    from sqlalchemy import create_engine, text, String, Text
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

try:
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    MISSING_DEPENDENCIES.append('pgvector')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from modules.phase4.__init__ import DATABASE_URL


class Base(DeclarativeBase):
    """Base class for vector-enabled models."""
    pass


class VectorDocument(Base):
    """Document model with vector embeddings."""
    __tablename__ = "vector_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 1536 dimensions for OpenAI text-embedding-3-small
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536))

    def __repr__(self) -> str:
        return f"<VectorDocument(id={self.id}, title='{self.title}')>"


def enable_pgvector() -> bool:
    """Enable the pgvector extension in PostgreSQL."""
    engine = create_engine(DATABASE_URL, echo=True)

    try:
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("\n✓ pgvector extension enabled")

            # Verify installation
            result = conn.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector'")
            )
            if result.rowcount > 0:
                print("✓ pgvector is active")
                return True
            else:
                print("✗ pgvector not found")
                return False

    except Exception as e:
        print(f"\n✗ Error enabling pgvector: {e}")
        print("\nMake sure you're using PostgreSQL with pgvector installed:")
        print("  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres \\")
        print("    ankane/pgvector")
        return False


def create_vector_tables() -> bool:
    """Create tables with vector columns."""
    engine = create_engine(DATABASE_URL, echo=True)

    try:
        Base.metadata.create_all(engine)
        print("\n✓ Vector tables created successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error creating tables: {e}")
        return False


def create_vector_index() -> bool:
    """Create an IVFFlat index for faster vector search."""
    engine = create_engine(DATABASE_URL, echo=True)

    try:
        with engine.connect() as conn:
            # Create IVFFlat index for cosine distance
            # lists=100 is good for up to 1M rows
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS vector_documents_embedding_idx
                ON vector_documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            conn.commit()
            print("\n✓ Vector index created")
            print("  Type: IVFFlat")
            print("  Distance: Cosine")
            print("  Lists: 100")
            return True

    except Exception as e:
        print(f"\n✗ Error creating index: {e}")
        return False


def show_vector_info() -> None:
    """Display information about vector setup."""
    engine = create_engine(DATABASE_URL, echo=False)

    with engine.connect() as conn:
        # Check pgvector version
        result = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        version = result.scalar()
        print(f"pgvector version: {version}")

        # Show vector columns
        result = conn.execute(text("""
            SELECT column_name, udt_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'vector_documents' AND udt_name = 'vector'
        """))
        print("\nVector columns:")
        for row in result:
            print(f"  - {row[0]}: vector({row[2]})")

        # Show indexes
        result = conn.execute(text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'vector_documents' AND indexname LIKE '%vector%'
        """))
        print("\nVector indexes:")
        for row in result:
            print(f"  - {row[0]}")


if __name__ == "__main__":
    print("pgvector Setup Demo")
    print("=" * 50)
    print(f"Database: {DATABASE_URL}")
    print("\nNote: Use PostgreSQL with pgvector:")
    print("  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres \\")
    print("    ankane/pgvector")
    print()

    # Enable extension
    print("\n1. ENABLE PGVECTOR EXTENSION")
    if not enable_pgvector():
        exit(1)

    # Create tables
    print("\n2. CREATE VECTOR TABLES")
    if not create_vector_tables():
        exit(1)

    # Create index
    print("\n3. CREATE VECTOR INDEX")
    if not create_vector_index():
        print("Warning: Index creation failed, but continuing...")

    # Show info
    print("\n4. VECTOR SETUP INFO")
    show_vector_info()

    print("\n" + "=" * 50)
    print("✓ pgvector setup complete!")
    print("\nNext: Run 05_vector_search_pg.py for semantic search")
