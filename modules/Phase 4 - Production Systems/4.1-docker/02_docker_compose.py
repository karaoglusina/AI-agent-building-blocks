"""
02 - Docker Compose
===================
Multi-container setup for AI applications.

Key concept: Docker Compose orchestrates multiple services - app, database, Redis, workers - as one system.

Book reference: —
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


def explain_docker_compose():
    """Explain Docker Compose for AI applications."""
    print("=== DOCKER COMPOSE FOR AI SYSTEMS ===\n")

    print("Typical AI application stack:\n")

    services = [
        "1. App: FastAPI application (main service)",
        "2. PostgreSQL: Database with pgvector extension",
        "3. Redis: Cache + Celery broker",
        "4. Celery Worker: Background job processing",
        "5. ChromaDB: Vector database (optional)"
    ]

    for service in services:
        print(service)

    print("\n" + "=" * 70)
    print("\nKey Docker Compose concepts:\n")

    concepts = [
        "- services: Define each container",
        "- depends_on: Service startup order",
        "- environment: Pass env variables",
        "- volumes: Persist data between restarts",
        "- networks: Services can talk to each other",
        "- ports: Expose to host machine",
        "- healthcheck: Verify service is working"
    ]

    for concept in concepts:
        print(concept)


def compose_commands():
    """Show common docker-compose commands."""
    print("\n" + "=" * 70)
    print("=== DOCKER COMPOSE COMMANDS ===\n")

    commands = [
        "# Start all services",
        "docker-compose up -d",
        "",
        "# View logs",
        "docker-compose logs -f app",
        "",
        "# Stop all services",
        "docker-compose down",
        "",
        "# Rebuild and restart",
        "docker-compose up -d --build",
        "",
        "# View running services",
        "docker-compose ps",
        "",
        "# Execute command in container",
        "docker-compose exec app python manage.py migrate",
        "",
        "# View logs for specific service",
        "docker-compose logs postgres",
        "",
        "# Scale service (multiple workers)",
        "docker-compose up -d --scale celery_worker=3"
    ]

    for cmd in commands:
        print(cmd)


def networking_explained():
    """Explain Docker networking."""
    print("\n" + "=" * 70)
    print("=== DOCKER NETWORKING ===\n")

    print("Services communicate using service names as hostnames:\n")

    examples = [
        "From app container:",
        "  - DATABASE_URL=postgresql://user:pass@postgres:5432/db",
        "  - REDIS_URL=redis://redis:6379/0",
        "",
        "From outside (host machine):",
        "  - localhost:8000 → app service",
        "  - localhost:5432 → postgres service",
        "  - localhost:6379 → redis service"
    ]

    for example in examples:
        print(example)


def env_file_example():
    """Show .env file example."""
    print("\n" + "=" * 70)
    print("=== ENVIRONMENT VARIABLES (.env) ===\n")

    env_content = """# OpenAI
OPENAI_API_KEY=sk-...

# Database
POSTGRES_USER=user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=aidb

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# App Config
ENV=production
DEBUG=false
"""

    print(env_content)
    print("\nLoad with: docker-compose --env-file .env up -d")


if __name__ == "__main__":
    explain_docker_compose()
    compose_commands()
    networking_explained()
    env_file_example()

    print("\n" + "=" * 70)
    print("\nKey insight: Docker Compose = orchestrate complex AI systems")
    print("\nSee docker-compose.yml example in this directory!")
