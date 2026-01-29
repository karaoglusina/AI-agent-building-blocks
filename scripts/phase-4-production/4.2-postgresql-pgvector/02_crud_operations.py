"""
02 - CRUD Operations
====================
Create, read, update, delete operations with SQLAlchemy.

Key concept: CRUD operations form the foundation of database interactions
in any data-driven application.

Book reference: —
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

from typing import Optional
try:
    from sqlalchemy import create_engine, select, update, delete
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

try:
    from sqlalchemy.orm import Session
except ImportError:
    MISSING_DEPENDENCIES.append('sqlalchemy')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from modules.phase4.__init__ import DATABASE_URL
from modules.phase4.__init__ import Document


def create_document(title: str, content: str, source: Optional[str] = None) -> Document:
    """Create a new document."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        doc = Document(title=title, content=content, source=source)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        print(f"✓ Created: {doc}")
        return doc


def read_document(doc_id: int) -> Optional[Document]:
    """Read a document by ID."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if doc:
            print(f"✓ Found: {doc}")
            print(f"  Content: {doc.content[:50]}...")
        else:
            print(f"✗ Document {doc_id} not found")
        return doc


def list_documents(limit: int = 10) -> list[Document]:
    """List all documents."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        stmt = select(Document).limit(limit)
        docs = list(session.scalars(stmt))
        print(f"✓ Found {len(docs)} documents:")
        for doc in docs:
            print(f"  - {doc}")
        return docs


def update_document(doc_id: int, **kwargs) -> bool:
    """Update a document's fields."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        stmt = update(Document).where(Document.id == doc_id).values(**kwargs)
        result = session.execute(stmt)
        session.commit()

        if result.rowcount > 0:
            print(f"✓ Updated document {doc_id}")
            return True
        else:
            print(f"✗ Document {doc_id} not found")
            return False


def delete_document(doc_id: int) -> bool:
    """Delete a document by ID."""
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:
        stmt = delete(Document).where(Document.id == doc_id)
        result = session.execute(stmt)
        session.commit()

        if result.rowcount > 0:
            print(f"✓ Deleted document {doc_id}")
            return True
        else:
            print(f"✗ Document {doc_id} not found")
            return False


if __name__ == "__main__":
    print("CRUD Operations Demo")
    print("=" * 50)

    # Create
    print("\n1. CREATE")
    doc = create_document(
        title="Deep Learning",
        content="Deep learning uses multi-layer neural networks.",
        source="dl_intro.txt"
    )

    # Read
    print("\n2. READ")
    read_document(doc.id)

    # List
    print("\n3. LIST")
    list_documents()

    # Update
    print("\n4. UPDATE")
    update_document(doc.id, content="Deep learning is a subset of machine learning.")

    # Read updated
    print("\n5. READ UPDATED")
    read_document(doc.id)

    # Delete
    print("\n6. DELETE")
    delete_document(doc.id)

    # Verify deletion
    print("\n7. VERIFY DELETION")
    read_document(doc.id)
