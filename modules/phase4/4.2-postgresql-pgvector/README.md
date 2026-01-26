# Module 4.2: PostgreSQL + pgvector

> *"Store and search embeddings at scale with PostgreSQL"*

This module covers using PostgreSQL with the pgvector extension for production-grade vector storage, combining relational data with semantic search capabilities.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_sqlalchemy_basics.py` | SQLAlchemy Basics | Define models, create tables with ORM |
| `02_crud_operations.py` | CRUD Operations | Create, read, update, delete with SQLAlchemy |
| `03_alembic_migrations.py` | Alembic Migrations | Database schema versioning and migrations |
| `04_pgvector_setup.py` | pgvector Setup | Enable vector extension, create indexes |
| `05_vector_search_pg.py` | Vector Search | Semantic search with cosine similarity |
| `06_hybrid_pg.py` | Hybrid Search | Combine full-text + vector search |

## Why PostgreSQL + pgvector?

PostgreSQL with pgvector offers powerful advantages for AI applications:

### Advantages
- **Single Database**: No need for separate vector databases
- **ACID Compliance**: Transactions, consistency, reliability
- **Mature Ecosystem**: Battle-tested with extensive tooling
- **Rich Features**: Joins, constraints, triggers, full-text search
- **Cost Effective**: Use existing PostgreSQL infrastructure
- **Hybrid Search**: Combine vector + keyword + metadata filters

### When to Use
- Building production AI applications
- Need transactional guarantees
- Complex queries combining vectors and relational data
- Already using PostgreSQL
- Hybrid search requirements
- Cost-sensitive projects

## Core Concepts

### 1. pgvector Extension
Adds vector data type and similarity operations to PostgreSQL:

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)  -- OpenAI embedding size
);
```

### 2. Vector Operations

**Distance Functions**:
- `<->` Euclidean (L2) distance
- `<#>` Negative inner product
- `<=>` Cosine distance (most common for embeddings)

```sql
-- Find nearest neighbors
SELECT * FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

### 3. Vector Indexes

**IVFFlat Index** (faster, approximate):
```sql
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**HNSW Index** (faster, more accurate):
```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops);
```

### 4. SQLAlchemy Integration

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import mapped_column

class Document(Base):
    embedding: Mapped[Vector] = mapped_column(Vector(1536))
```

## Prerequisites

### Install PostgreSQL with pgvector

**Option 1: Docker (Recommended)**
```bash
# Run PostgreSQL with pgvector
docker run -d \
  --name postgres-pgvector \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agent_db \
  ankane/pgvector

# Or with docker-compose
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  postgres:
    image: ankane/pgvector
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: agent_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF

docker-compose up -d
```

**Option 2: Install Extension on Existing PostgreSQL**
```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS (Homebrew)
brew install pgvector

# Then in PostgreSQL
CREATE EXTENSION vector;
```

### Install Python Dependencies

```bash
# Using pip
pip install sqlalchemy psycopg2-binary pgvector alembic openai

# Using uv (recommended)
uv pip install sqlalchemy psycopg2-binary pgvector alembic openai
```

### Environment Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Set database URL (if different from default)
export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'
```

## Running the Examples

### 1. Basic SQLAlchemy Setup
```bash
python 01_sqlalchemy_basics.py
```
Creates tables and inserts sample data.

### 2. CRUD Operations
```bash
python 02_crud_operations.py
```
Demonstrates create, read, update, delete operations.

### 3. Database Migrations
```bash
python 03_alembic_migrations.py
```
Shows Alembic workflow (run commands manually).

### 4. Enable pgvector
```bash
python 04_pgvector_setup.py
```
Enables pgvector extension and creates vector tables.

### 5. Vector Search
```bash
python 05_vector_search_pg.py
```
Semantic search using vector embeddings.

### 6. Hybrid Search
```bash
python 06_hybrid_pg.py
```
Combines full-text and vector search.

## Database Schema

### Documents Table (Basic)
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Vector Documents Table
```sql
CREATE TABLE vector_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    content_tsv TSVECTOR  -- For full-text search
);

-- Vector index (IVFFlat)
CREATE INDEX ON vector_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search index
CREATE INDEX ON vector_documents
USING GIN(content_tsv);
```

## SQLAlchemy Patterns

### Define Models
```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    embedding: Mapped[Vector] = mapped_column(Vector(1536))
```

### Create Engine and Session
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql://user:pass@localhost/db")

with Session(engine) as session:
    # Your operations here
    session.add(doc)
    session.commit()
```

### Query Examples
```python
# Select all
docs = session.execute(select(Document)).scalars().all()

# Filter
docs = session.execute(
    select(Document).where(Document.title == "AI")
).scalars().all()

# Vector search
results = session.execute(
    select(Document)
    .order_by(Document.embedding.cosine_distance(query_vec))
    .limit(5)
).scalars().all()
```

## Alembic Migrations

### Initialize Alembic
```bash
cd modules/phase4/4.2-postgresql-pgvector
alembic init alembic
```

### Configure alembic.ini
```ini
sqlalchemy.url = postgresql://postgres:postgres@localhost:5432/agent_db
```

### Configure env.py
```python
from modules.phase4 import Base

target_metadata = Base.metadata
```

### Create Migration
```bash
# Auto-generate from models
alembic revision --autogenerate -m "Add documents table"

# Manual migration
alembic revision -m "Add index"
```

### Apply Migrations
```bash
# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# View current version
alembic current

# View history
alembic history
```

## Vector Search Strategies

### 1. Basic Similarity Search
```python
# Find similar documents
query_embedding = get_embedding("machine learning")
results = session.execute(
    select(Document)
    .order_by(Document.embedding.cosine_distance(query_embedding))
    .limit(5)
).scalars().all()
```

### 2. With Filters
```python
# Combine vector search with filters
results = session.execute(
    select(Document)
    .where(Document.source == "research")
    .order_by(Document.embedding.cosine_distance(query_embedding))
    .limit(5)
).scalars().all()
```

### 3. Distance Threshold
```python
# Only return documents within distance threshold
results = session.execute(
    select(Document)
    .where(Document.embedding.cosine_distance(query_embedding) < 0.5)
    .order_by(Document.embedding.cosine_distance(query_embedding))
).scalars().all()
```

### 4. Hybrid Search
```python
# Combine full-text and vector search
vector_results = vector_search(query)
keyword_results = keyword_search(query)

# Weighted combination
combined_score = (
    vector_weight * vector_score +
    keyword_weight * keyword_score
)
```

## Index Types

### IVFFlat Index
- **Type**: Approximate nearest neighbor (ANN)
- **Speed**: Fast queries, slower builds
- **Accuracy**: Good (configurable via lists)
- **Use Case**: Large datasets (>10K vectors)

```sql
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust lists based on row count
```

**Lists Parameter**:
- Fewer lists = better recall, slower search
- More lists = faster search, lower recall
- Rule of thumb: `lists = rows / 1000`

### HNSW Index
- **Type**: Approximate nearest neighbor (ANN)
- **Speed**: Very fast queries, slower builds
- **Accuracy**: Excellent
- **Use Case**: Read-heavy workloads

```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### When to Use Indexes
- **<10K vectors**: No index needed (fast enough)
- **10K-1M vectors**: IVFFlat recommended
- **>1M vectors**: HNSW or IVFFlat with tuning

## Performance Optimization

### 1. Index Configuration
```sql
-- IVFFlat: Adjust lists
-- rows < 100K: lists = 100
-- rows > 100K: lists = rows / 1000

-- HNSW: Adjust m and ef_construction
-- Higher values = better recall, slower build
```

### 2. Connection Pooling
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10
)
```

### 3. Batch Operations
```python
# Insert in batches
session.bulk_insert_mappings(Document, batch_data)
session.commit()
```

### 4. Analyze Tables
```sql
ANALYZE documents;  -- Update statistics
VACUUM ANALYZE documents;  -- Reclaim space + analyze
```

## Best Practices

### Schema Design
1. **Normalize relational data**: Use proper foreign keys
2. **Denormalize for search**: Include searchable text in vector table
3. **Partition large tables**: By date or category
4. **Use appropriate types**: TEXT vs VARCHAR, JSONB for metadata

### Vector Storage
1. **Dimension consistency**: All vectors same size
2. **Normalize embeddings**: Unit vectors for cosine similarity
3. **Index after bulk insert**: Create index after loading data
4. **Monitor index usage**: Use `pg_stat_user_indexes`

### Query Optimization
1. **Use indexes**: Create appropriate vector indexes
2. **Limit results**: Always use LIMIT in production
3. **Filter before search**: Use WHERE clauses efficiently
4. **Explain plans**: Use EXPLAIN ANALYZE

### Production Readiness
1. **Connection pooling**: Use pgBouncer or SQLAlchemy pool
2. **Monitoring**: Track query performance, index health
3. **Backups**: Regular pg_dump or continuous archiving
4. **Replication**: Read replicas for scaling reads

## Common Issues & Solutions

### Issue: Slow Vector Search
**Solution**: Create appropriate index
```sql
-- For IVFFlat, train on sample data first
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Issue: Out of Memory
**Solution**: Increase work_mem or adjust query
```sql
SET work_mem = '256MB';
```

### Issue: Index Not Used
**Solution**: Analyze table and check query
```sql
ANALYZE documents;
EXPLAIN ANALYZE
SELECT * FROM documents
ORDER BY embedding <=> '[...]'
LIMIT 10;
```

### Issue: Slow Inserts with Index
**Solution**: Drop index, bulk insert, recreate index
```sql
DROP INDEX documents_embedding_idx;
-- Bulk insert here
CREATE INDEX ...;
```

## Security Considerations

### 1. Connection Security
```python
# Use SSL in production
DATABASE_URL = "postgresql://user:pass@host/db?sslmode=require"
```

### 2. Credentials Management
```python
# Never hardcode credentials
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"
```

### 3. SQL Injection Prevention
```python
# Always use parameterized queries
session.execute(
    select(Document).where(Document.id == user_input)  # Safe
)
# Never: f"SELECT * FROM documents WHERE id = {user_input}"
```

### 4. Access Control
```sql
-- Create read-only user for app
CREATE USER app_readonly WITH PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
```

## Monitoring & Maintenance

### Key Metrics
```sql
-- Table size
SELECT pg_size_pretty(pg_total_relation_size('documents'));

-- Index size
SELECT pg_size_pretty(pg_relation_size('documents_embedding_idx'));

-- Index usage
SELECT * FROM pg_stat_user_indexes WHERE relname = 'documents';

-- Slow queries
SELECT * FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

### Maintenance Tasks
```bash
# Regular vacuum
VACUUM ANALYZE documents;

# Reindex periodically
REINDEX INDEX documents_embedding_idx;

# Update statistics
ANALYZE documents;
```

## Comparison: pgvector vs. Vector Databases

| Feature | pgvector | Pinecone/Weaviate |
|---------|----------|-------------------|
| Setup | Extension on PostgreSQL | Separate service |
| Cost | Free (hosting only) | Subscription |
| Scalability | Good (TB scale) | Excellent (PB scale) |
| Features | SQL + vectors | Vectors only |
| Transactions | ACID compliant | Eventually consistent |
| Maintenance | Standard PostgreSQL | Managed service |
| Best For | Most applications | Massive vector-only workloads |

## Book References

- `AI_eng.6` - Vector databases and semantic search

## Next Steps

After mastering PostgreSQL + pgvector:
- Module 4.3: Observability with Langfuse (track vector searches)
- Module 4.5: Async & Background Jobs (async database operations)
- Module 4.7: Cloud Deployment (managed PostgreSQL, RDS)

## Additional Resources

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
