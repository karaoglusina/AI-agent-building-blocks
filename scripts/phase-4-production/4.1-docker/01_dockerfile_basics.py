"""
01 - Dockerfile Basics
=======================
Create Dockerfile for Python AI application.

Key concept: Dockerfiles define reproducible environments - same code runs identically everywhere.

Book reference: —
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def explain_dockerfile():
    """Explain Dockerfile structure for AI applications."""
    print("=== DOCKERFILE FOR AI APPLICATIONS ===\n")

    print("Basic structure:\n")

    dockerfile_content = """# Start from official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (if running API)
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
"""

    print(dockerfile_content)

    print("\n=== KEY CONCEPTS ===\n")

    concepts = [
        "1. Layer Caching: Copy requirements before code (cache dependencies)",
        "2. Slim Images: Use -slim or -alpine for smaller images",
        "3. .dockerignore: Exclude .git, __pycache__, .env, venv/",
        "4. Multi-stage: Build dependencies in one stage, copy to clean image",
        "5. Non-root User: Run as non-root for security"
    ]

    for concept in concepts:
        print(concept)


def production_dockerfile():
    """Show production-ready Dockerfile."""
    print("\n" + "=" * 70)
    print("=== PRODUCTION DOCKERFILE ===\n")

    prod_dockerfile = """FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---

FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    print(prod_dockerfile)


def docker_commands():
    """Show common Docker commands."""
    print("\n" + "=" * 70)
    print("=== COMMON DOCKER COMMANDS ===\n")

    commands = [
        "# Build image",
        "docker build -t my-ai-app:latest .",
        "",
        "# Run container",
        "docker run -p 8000:8000 --env-file .env my-ai-app:latest",
        "",
        "# Run interactively",
        "docker run -it my-ai-app:latest /bin/bash",
        "",
        "# View logs",
        "docker logs <container-id>",
        "",
        "# List containers",
        "docker ps",
        "",
        "# Stop container",
        "docker stop <container-id>",
        "",
        "# Remove image",
        "docker rmi my-ai-app:latest"
    ]

    for cmd in commands:
        print(cmd)


if __name__ == "__main__":
    explain_dockerfile()
    production_dockerfile()
    docker_commands()

    print("\n" + "=" * 70)
    print("\nKey insight: Docker = reproducible, portable AI deployments")
    print("\nSee Dockerfile example in this directory!")
