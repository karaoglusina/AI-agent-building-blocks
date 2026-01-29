"""
Phase 4: Production-Ready AI Systems

Shared models and configuration for PostgreSQL modules.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


# Database URL - use environment variable in production
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/agent_db"


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
