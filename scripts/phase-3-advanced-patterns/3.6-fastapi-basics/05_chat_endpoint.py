"""
05 - Chat API Endpoint
=======================
Full chat endpoint with conversation history.

Key concept: Stateful conversation requires managing message history - either client-side or server-side storage.

Book reference: NLP_cook.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import utils._load_env  # Loads .env file automatically
try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    MISSING_DEPENDENCIES.append('fastapi')

from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import Optional
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

app = FastAPI()
client = AsyncOpenAI()

# In-memory storage (use Redis/DB in production)
conversations: dict[str, list[dict]] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    system_prompt: Optional[str] = "You are a helpful job search assistant."


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    message_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation history.

    If conversation_id is provided, continues that conversation.
    Otherwise, starts a new conversation.
    """
    # Get or create conversation
    conv_id = request.conversation_id or f"conv_{len(conversations)}"

    if conv_id not in conversations:
        # New conversation
        conversations[conv_id] = [
            {"role": "system", "content": request.system_prompt}
        ]

    # Add user message
    conversations[conv_id].append({
        "role": "user",
        "content": request.message
    })

    # Get response from LLM
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations[conv_id],
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response to history
        conversations[conv_id].append({
            "role": "assistant",
            "content": assistant_message
        })

        return ChatResponse(
            conversation_id=conv_id,
            message=assistant_message,
            message_count=len(conversations[conv_id]) - 1  # Exclude system message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{conversation_id}/history")
async def get_history(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Return all messages except system prompt
    messages = conversations[conversation_id][1:]  # Skip system message

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "count": len(messages)
    }


@app.delete("/chat/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/conversations")
async def list_conversations():
    """List all active conversations."""
    return {
        "conversations": [
            {
                "id": conv_id,
                "message_count": len(messages) - 1  # Exclude system
            }
            for conv_id, messages in conversations.items()
        ],
        "total": len(conversations)
    }


if __name__ == "__main__":
    import uvicorn
    import os

    print("Starting Chat API...")
    print("\nEndpoints:")
    print("  POST /chat - Send message (with optional conversation_id)")
    print("  GET /chat/{id}/history - Get conversation history")
    print("  DELETE /chat/{id} - Delete conversation")
    print("  GET /conversations - List all conversations")
    print("\nExample conversation:")
    print('  # Start new conversation')
    print('  curl -X POST "http://localhost:8000/chat" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"message": "Hello!"}\'')
    print('\n  # Continue conversation (use returned conversation_id)')
    print('  curl -X POST "http://localhost:8000/chat" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"conversation_id": "conv_0", "message": "What jobs are available?"}\'')

    # Skip server startup in test mode
    if os.getenv("TEST_MODE") == "1":
        print("Test mode: Skipping server startup")
        print("FastAPI app created successfully")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
