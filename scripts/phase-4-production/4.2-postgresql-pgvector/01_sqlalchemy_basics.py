"""
01 - SQLAlchemy Basics
======================
Define models, create tables with SQLAlchemy ORM.

Key concept: SQLAlchemy provides a Pythonic way to interact with databases
using declarative models and type-safe queries.

Book reference: —
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

from datetime import datetime
from typing import Optional
try:
    from sqlalchemy import create_engine, String, Text, DateTime, Integer
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

try:
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


# Database URL - use environment variable in production
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/agent_db"


# Base class for declarative models
class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Document(Base):
    """Document model for storing text content."""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}')>"


def create_tables() -> None:
    """Create all tables in the database."""
    engine = create_engine(DATABASE_URL, echo=True)
    Base.metadata.create_all(engine)
    print("\n✓ Tables created successfully!")


def insert_sample_data() -> None:
    """Insert sample documents into the database."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        docs = [
            Document(
                title="Introduction to AI",
                content="Artificial Intelligence is transforming technology.",
                source="ai_basics.txt"
            ),
            Document(
                title="Machine Learning Fundamentals",
                content="ML algorithms learn patterns from data.",
                source="ml_guide.txt"
            ),
            Document(
                title="Neural Networks",
                content="Neural networks mimic the human brain structure.",
                source="nn_intro.txt"
            )]

        session.add_all(docs)
        session.commit()
        print(f"\n✓ Inserted {len(docs)} sample documents")


if __name__ == "__main__":
    print("SQLAlchemy Basics Demo")
    print("=" * 50)
    print(f"Database: {DATABASE_URL}")
    print("\nNote: Ensure PostgreSQL is running:")
    print("  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16")
    print()

    try:
        create_tables()
        insert_sample_data()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure PostgreSQL is running and accessible.")
