"""
06 - RAG API Endpoint
======================
Complete RAG endpoint: query → retrieve → generate.

Key concept: Production RAG needs proper API structure with search, context assembly, and response generation.

Book reference: AI_eng.6
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

app = FastAPI()
client = AsyncOpenAI()

# Initialize ChromaDB
chroma_client = chromadb.Client()


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class RAGRequest(BaseModel):
    query: str
    n_results: int = 3
    include_sources: bool = True


class SearchResult(BaseModel):
    id: str
    content: str
    similarity: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[list[SearchResult]] = None
    context_used: int


def initialize_collection():
    """Initialize job search collection (call once)."""
    try:
        collection = chroma_client.get_collection("jobs")
        print(f"Collection exists with {collection.count()} documents")
        return collection
    except:
        print("Creating new collection and indexing jobs...")
        collection = chroma_client.create_collection("jobs")

        # Index sample jobs
        jobs = load_sample_jobs(100)

        documents = []
        metadatas = []
        ids = []

        for i, job in enumerate(jobs):
            doc = f"Title: {job['title']}\nCompany: {job.get('company', 'Unknown')}\nLocation: {job.get('location', 'Remote')}\n\n{job.get('description', '')[:500]}"
            documents.append(doc)
            metadatas.append({
                "title": job['title'],
                "company": job.get('company', 'Unknown'),
                "id": str(job.get('id', i))
            })
            ids.append(f"job_{i}")

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Indexed {len(documents)} jobs")
        return collection


# Initialize on startup
collection = initialize_collection()


@app.post("/search", response_model=list[SearchResult])
async def search_jobs(request: SearchRequest):
    """
    Search for relevant jobs using semantic search.

    Returns top-k most relevant job postings.
    """
    results = collection.query(
        query_texts=[request.query],
        n_results=request.n_results
    )

    search_results = []
    for i in range(len(results['ids'][0])):
        search_results.append(SearchResult(
            id=results['ids'][0][i],
            content=results['documents'][0][i],
            similarity=1 - results['distances'][0][i]  # Convert distance to similarity
        ))

    return search_results


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    RAG endpoint: retrieves relevant jobs and generates answer.

    Complete pipeline:
    1. Semantic search for relevant jobs
    2. Assemble context from retrieved docs
    3. Generate answer using LLM
    4. Optionally include sources
    """
    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[request.query],
        n_results=request.n_results
    )

    if not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No relevant jobs found")

    # Step 2: Assemble context
    context_parts = []
    for i, doc in enumerate(results['documents'][0]):
        context_parts.append(f"[Job {i+1}]\n{doc}")

    context = "\n\n".join(context_parts)

    # Step 3: Generate answer
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a job search assistant. Answer questions based on the provided job postings. Be specific and cite job titles when relevant."
                },
                {
                    "role": "user",
                    "content": f"""Context (relevant jobs):
{context}

Question: {request.query}

Answer:"""
                }
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content

        # Step 4: Prepare response
        sources = None
        if request.include_sources:
            sources = [
                SearchResult(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i][:200] + "...",
                    similarity=1 - results['distances'][0][i]
                )
                for i in range(len(results['ids'][0]))
            ]

        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            context_used=len(results['documents'][0])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with collection stats."""
    return {
        "status": "healthy",
        "collection_size": collection.count()
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting RAG API...")
    print(f"Indexed {collection.count()} jobs")
    print("\nEndpoints:")
    print("  POST /search - Semantic search only")
    print("  POST /rag - Full RAG (retrieve + generate)")
    print("  GET /health - Health check")
    print("\nExample RAG query:")
    print('  curl -X POST "http://localhost:8000/rag" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What Python jobs are available?", "n_results": 3}\'')
    print("\nRAG = Retrieval + Generation = Grounded answers!")

    uvicorn.run(app, host="0.0.0.0", port=8000)
