"""
01 - Hello FastAPI
==================
Minimal API setup with basic routes and automatic docs.

Key concept: FastAPI provides automatic OpenAPI docs and type validation out of the box.

Book reference: —
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    from fastapi import FastAPI
except ImportError:
    MISSING_DEPENDENCIES.append('fastapi')

try:
    from fastapi.responses import JSONResponse
except ImportError:
    MISSING_DEPENDENCIES.append('fastapi')

try:
    import uvicorn
except ImportError:
    MISSING_DEPENDENCIES.append('uvicorn')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

# Create FastAPI application
app = FastAPI(
    title="Hello FastAPI",
    description="A minimal FastAPI application for AI agents",
    version="1.0.0"
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint - health check."""
    return {"message": "Hello from FastAPI!", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "fastapi-basics"}


@app.get("/info")
def get_info() -> dict[str, str | int]:
    """Get API information."""
    return {
        "name": "Hello FastAPI",
        "version": "1.0.0",
        "endpoints": 4,
        "framework": "FastAPI"
    }


@app.get("/echo/{message}")
def echo_message(message: str) -> dict[str, str]:
    """Echo back a message from path parameter."""
    return {"original": message, "echo": f"You said: {message}"}


@app.get("/greet")
def greet_user(name: str = "World", greeting: str = "Hello") -> dict[str, str]:
    """Greet a user with query parameters."""
    return {"greeting": f"{greeting}, {name}!"}


if __name__ == "__main__":
    import os
    # Skip server startup in test mode
    if os.getenv("TEST_MODE") == "1":
        print("Test mode: Skipping server startup")
        print("FastAPI app created successfully")
    else:
        print("Starting FastAPI server...")
        print("📚 API docs available at: http://localhost:8000/docs")
        print("📖 Alternative docs at: http://localhost:8000/redoc")
        print("\nExample requests:")
        print("  curl http://localhost:8000/")
        print("  curl http://localhost:8000/echo/test")
        print("  curl http://localhost:8000/greet?name=Claude")
        print("\nPress CTRL+C to stop\n")

        uvicorn.run(app, host="0.0.0.0", port=8000)
