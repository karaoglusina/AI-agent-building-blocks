"""
04 - Streaming Responses
=========================
Stream LLM output to client for better UX.

Key concept: Don't wait for full completion - stream tokens as they're generated for responsive feel.

Book reference: AI_eng.9
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

app = FastAPI()
client = AsyncOpenAI()


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."


async def generate_stream(message: str, system_prompt: str):
    """Generate streaming response from OpenAI."""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        stream=True  # Enable streaming
    )

    # Yield chunks as they arrive
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # Send as Server-Sent Events format
            yield f"data: {content}\n\n"

    # Signal completion
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat completions to client.

    Returns Server-Sent Events (SSE) format.
    """
    return StreamingResponse(
        generate_stream(request.message, request.system_prompt),
        media_type="text/event-stream"
    )


@app.post("/chat/complete")
async def chat_complete(request: ChatRequest):
    """Non-streaming endpoint for comparison."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ],
        temperature=0.7
    )

    return {
        "message": response.choices[0].message.content,
        "streaming": False
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI with streaming support...")
    print("\nEndpoints:")
    print("  POST /chat/stream - Stream response (SSE)")
    print("  POST /chat/complete - Complete response (no streaming)")
    print("\nTry streaming:")
    print('  curl -X POST "http://localhost:8000/chat/stream" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"message": "Write a short poem about AI"}\'')
    print("\nStreaming = better UX for long responses!")

    uvicorn.run(app, host="0.0.0.0", port=8000)
