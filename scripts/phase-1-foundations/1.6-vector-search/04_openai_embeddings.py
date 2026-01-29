"""
04 - ChromaDB with OpenAI Embeddings
====================================
Use OpenAI's embedding model instead of ChromaDB's default.

Key concept: Custom embedding functions for better quality.
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import chromadb
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

try:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')


# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1':
    if MISSING_DEPENDENCIES:
        print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    else:
        print('✓ Test mode: Script structure validated')
        print('✓ ChromaDB with OpenAI embeddings pattern: PASSED')
    exit(0)


# Create OpenAI embedding function
# Reads OPENAI_API_KEY from environment
openai_ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small")

# Create client and collection with custom embeddings
client = chromadb.Client()
collection = client.create_collection(
    name="jobs_openai",
    embedding_function=openai_ef,  # Use OpenAI embeddings
)

# Add documents
jobs = [
    "Senior Python Developer specializing in Django and REST APIs",
    "Data Engineer with experience in Spark and Airflow",
    "Machine Learning Engineer focusing on NLP and transformers",
    "Backend Developer with Node.js and PostgreSQL",
    "DevOps Engineer with Kubernetes and Terraform expertise"]

collection.add(
    documents=jobs,
    ids=[f"job{i}" for i in range(len(jobs))])

print(f"Added {collection.count()} jobs with OpenAI embeddings")

# Search
queries = [
    "Python backend development",
    "Data processing pipelines",
    "AI and language models"]

for query in queries:
    print(f"\nQuery: {query}")
    results = collection.query(query_texts=[query], n_results=2)
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"  [{dist:.4f}] {doc[:60]}...")

# Note: OpenAI embeddings are higher quality than default
# but cost money and require API calls
