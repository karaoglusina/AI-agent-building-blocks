"""
Shared database models and configuration for Phase 4 (PostgreSQL + pgvector).

These live in `utils/` rather than next to the Phase 4 scripts for a boring reason:
`scripts/phase-4-production/` contains hyphens, so it can never be a Python package
and can never be imported. `utils` is already a package and is already on the path in
every script, so this is the one place the Phase 4 modules can genuinely share models.

Requires the phase-4 dependency group (`uv sync --group phase-4`).
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# Override with DATABASE_URL in your environment; the default matches the
# docker-compose service in module 4.1.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/agent_db",
)


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
