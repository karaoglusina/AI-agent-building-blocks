# Phase 4: Production

Building AI systems is one thing. Running them reliably in production is another. Phase 4 covers the infrastructure, monitoring, security, and deployment practices needed to go from prototype to product.

## What You're Building Toward

By the end of Phase 4, you'll be able to:
- Containerize your AI applications with Docker
- Use PostgreSQL with pgvector for production-grade vector storage
- Monitor and trace AI systems with observability tools
- Protect against attacks and misuse with guardrails
- Handle background jobs and async processing
- Work with MCP servers for tool integration
- Deploy to the cloud with proper infrastructure
- Set up CI/CD pipelines for automated deployment

This is where your job market analyzer becomes a real product that other people can use reliably.

## The Modules

### [Docker](./docker.md)
Containerizing your AI applications. Dockerfiles, Docker Compose, multi-service setups. The foundation for reproducible deployment.

### [PostgreSQL + pgvector](./postgresql-pgvector.md)
Production-grade vector storage. SQLAlchemy, migrations, hybrid search. When ChromaDB isn't enough.

### [Observability](./observability.md)
See what's happening in production. Tracing, cost monitoring, custom metrics. Langfuse for LLM-specific observability.

### [Guardrails](./guardrails.md)
Protect your system. Input validation, prompt injection defense, PII filtering, content moderation. Defense in depth.

### [Async & Background Jobs](./async-background-jobs.md)
Handle long-running tasks. Asyncio, Celery, task queues. Essential for processing that can't happen in request time.

### [MCP Servers](./mcp-servers.md)
Tool integration protocol. Connect to MCP servers, use MCP tools, build custom servers.

### [Cloud Deployment](./cloud-deployment.md)
Put it in the cloud. VM setup, HTTPS, health checks, logging. Getting from localhost to the internet.

### [CI/CD](./cicd.md)
Automate everything. GitHub Actions, automated tests, deployment workflows. Push code, get deployments.

## The Flow

Phase 4 modules are fairly independent:

```
Docker ─────────────────────────┐
                                ├── Cloud Deployment ── CI/CD
PostgreSQL + pgvector ──────────┤
                                │
Observability ──────────────────┤
                                │
Guardrails ─────────────────────┤
                                │
Async & Background Jobs ────────┘

MCP Servers (independent)
```

Start with Docker - it's the foundation for everything else. Then add what your application needs.

## What You'll Need

```bash
# Containers
# Install Docker Desktop from docker.com

# Database
pip install sqlalchemy psycopg2-binary pgvector alembic

# Observability
pip install langfuse

# Background jobs
pip install celery redis
```

## The Job Market Analyzer at Phase 4

With Phase 4 infrastructure, your system can:
- Run in Docker containers anywhere
- Store millions of job embeddings in PostgreSQL
- Monitor costs and trace every query
- Block malicious inputs and protect PII
- Process job data imports in background jobs
- Deploy automatically when you push code
- Scale to handle real user traffic
