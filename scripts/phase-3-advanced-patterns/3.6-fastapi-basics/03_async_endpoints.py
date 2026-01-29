"""
03 - Async Endpoints
=====================
Handle concurrent requests efficiently with async/await.

Key concept: Async endpoints don't block - server can handle multiple requests concurrently, improving throughput.

Book reference: AI_eng.9
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import utils._load_env  # Loads .env file automatically
try:
    from fastapi import FastAPI
except ImportError:
    MISSING_DEPENDENCIES.append('fastapi')

from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

app = FastAPI()
client = AsyncOpenAI()


class AnalysisRequest(BaseModel):
    text: str
    analyses: list[str]  # e.g., ["sentiment", "summary", "keywords"]


class AnalysisResponse(BaseModel):
    results: dict[str, str]
    processing_time: float


async def analyze_sentiment(text: str) -> str:
    """Async sentiment analysis."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze sentiment: positive, negative, or neutral. One word only."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()


async def analyze_summary(text: str) -> str:
    """Async summarization."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize in one sentence."},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


async def extract_keywords(text: str) -> str:
    """Async keyword extraction."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract top 5 keywords, comma-separated."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """
    Async endpoint that runs multiple analyses concurrently.

    Example request:
    {
        "text": "This product is amazing! I love it.",
        "analyses": ["sentiment", "summary", "keywords"]
    }
    """
    import time
    start = time.time()

    # Map analysis types to functions
    analysis_functions = {
        "sentiment": analyze_sentiment,
        "summary": analyze_summary,
        "keywords": extract_keywords
    }

    # Run all requested analyses concurrently
    tasks = []
    for analysis_type in request.analyses:
        if analysis_type in analysis_functions:
            tasks.append(analysis_functions[analysis_type](request.text))

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Combine results
    result_dict = {
        analysis_type: result
        for analysis_type, result in zip(request.analyses, results)
        if analysis_type in analysis_functions
    }

    elapsed = time.time() - start

    return AnalysisResponse(
        results=result_dict,
        processing_time=elapsed
    )


@app.get("/health")
async def health_check():
    """Async health check endpoint."""
    return {"status": "healthy", "async": True}


if __name__ == "__main__":
    import uvicorn
    import os

    print("Starting FastAPI with async endpoints...")
    print("Endpoints:")
    print("  POST /analyze - Run multiple analyses concurrently")
    print("  GET /health - Health check")
    print("\nTry:")
    print('  curl -X POST "http://localhost:8000/analyze" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"text": "This is great!", "analyses": ["sentiment", "summary", "keywords"]}\'')
    print("\nAsync = concurrent processing = faster responses!")

    # Skip server startup in test mode
    if os.getenv("TEST_MODE") == "1":
        print("Test mode: Skipping server startup")
        print("FastAPI app created successfully")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
