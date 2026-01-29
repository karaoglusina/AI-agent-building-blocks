# Module 3.6: FastAPI Basics

> *"Build APIs for your AI systems"*

This module covers building production-ready APIs for AI applications using FastAPI, including request validation, async endpoints, streaming, chat, and RAG APIs.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_hello_fastapi.py` | Hello FastAPI | Minimal API setup with automatic docs |
| `02_pydantic_validation.py` | Request/Response Validation | Use Pydantic models for type-safe APIs |
| `03_async_endpoints.py` | Async Endpoints | Non-blocking endpoints handle concurrent requests |
| `04_streaming_response.py` | Streaming Responses | Stream LLM output for better UX |
| `05_chat_endpoint.py` | Chat API | Stateful conversation with history management |
| `06_rag_endpoint.py` | RAG API | Complete RAG pipeline as API |

## Why FastAPI?

FastAPI is ideal for AI applications:
- **Fast**: High performance (on par with NodeJS)
- **Type-safe**: Pydantic validation catches errors early
- **Async**: Native async/await for concurrent processing
- **Auto docs**: OpenAPI/Swagger docs generated automatically
- **Modern**: Python 3.6+ features (type hints, async)

## Core Concepts

### 1. Basic API Structure
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}
```

### 2. Pydantic Validation
```python
class Request(BaseModel):
    text: str
    max_length: int = 100

@app.post("/endpoint")
def endpoint(request: Request):
    # Automatic validation!
    return process(request.text)
```

### 3. Async Endpoints
```python
@app.post("/async")
async def async_endpoint(request: Request):
    result = await async_function()
    return result
```

### 4. Streaming
```python
@app.post("/stream")
async def stream_endpoint():
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

## API Patterns for AI

### Chat API Pattern
```
Client → POST /chat
       ← conversation_id + response
       → POST /chat (with conversation_id)
       ← response (continues conversation)
```

### RAG API Pattern
```
Client → POST /rag {"query": "..."}
       ↓
       Search vector DB
       ↓
       Assemble context
       ↓
       Generate answer
       ↓
Client ← Answer + sources
```

### Streaming Pattern
```
Client → POST /chat/stream
       ← SSE: "data: token1\n\n"
       ← SSE: "data: token2\n\n"
       ← SSE: "data: token3\n\n"
       ← SSE: "data: [DONE]\n\n"
```

## Running the Scripts

Each script starts a server on `http://localhost:8000`:

```bash
python 01_hello_fastapi.py
# Visit http://localhost:8000/docs for interactive API docs
```

Test endpoints with curl:
```bash
# POST request
curl -X POST "http://localhost:8000/endpoint" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "max_length": 50}'

# GET request
curl "http://localhost:8000/health"
```

Or visit the auto-generated docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Prerequisites

Install required libraries:

```bash
pip install fastapi uvicorn pydantic openai chromadb
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Testing Your APIs

### Using curl
```bash
curl -X POST "http://localhost:8000/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python jobs in SF", "n_results": 3}'
```

### Using Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/rag",
    json={"query": "Python jobs", "n_results": 3}
)
print(response.json())
```

### Using the interactive docs
1. Start the server
2. Visit `http://localhost:8000/docs`
3. Click "Try it out" on any endpoint
4. Fill in parameters
5. Click "Execute"

## Production Considerations

### Environment Configuration
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    db_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### Error Handling
```python
from fastapi import HTTPException

@app.post("/endpoint")
async def endpoint(request: Request):
    try:
        result = await process(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: "global")

@app.post("/endpoint")
@limiter.limit("10/minute")
async def endpoint(request: Request):
    return process(request)
```

### CORS (for frontend)
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Tips

1. **Use async**: `async def` for I/O-bound operations
2. **Connection pooling**: Reuse DB/API clients
3. **Caching**: Cache embeddings, frequent queries
4. **Background tasks**: Long operations → Celery
5. **Streaming**: Stream long responses for better UX

## Common Patterns

| Use Case | Pattern |
|----------|---------|
| Simple LLM call | Sync endpoint |
| Multiple concurrent calls | Async endpoint |
| Long generation | Streaming |
| Conversation | Chat with history |
| Knowledge base | RAG endpoint |
| Heavy processing | Background task |

## Job Data Applications

These endpoints work great for:
- **Search API**: Semantic job search
- **Chat API**: Conversational job recommendations
- **RAG API**: Q&A over job database
- **Streaming**: Live job description generation
- **Async**: Batch job classification

## Book References

- `AI_eng.2` - Pydantic for data validation
- `AI_eng.6` - RAG architecture
- `AI_eng.9` - Async processing and optimization
- `NLP_cook.10` - Conversational agents

## Next Steps

After mastering FastAPI basics:
- Module 3.7: Clustering & Topics
- Module 3.8: Evaluation Systems
- Module 4.1: Docker & Containerization
- Module 4.5: Background Jobs with Celery
