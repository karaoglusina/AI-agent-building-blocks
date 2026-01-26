# Module 4.1: Docker & Containerization

> *"Package and deploy AI applications in containers"*

This module covers containerizing AI applications with Docker, orchestrating multi-service systems with Docker Compose, and managing configuration securely.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_dockerfile_basics.py` | Dockerfile Basics | Define reproducible environments for AI apps |
| `02_docker_compose.py` | Docker Compose | Orchestrate multi-container AI systems |
| `03_environment_config.py` | Environment Configuration | Secure configuration with environment variables |
| `Dockerfile` | Production Dockerfile | Multi-stage, optimized container image |
| `docker-compose.yml` | Full Stack Compose | App + PostgreSQL + Redis + Celery |
| `.dockerignore` | Docker Ignore | Exclude unnecessary files from image |

## Why Docker?

Docker solves deployment challenges:
- **Reproducibility**: Same environment everywhere
- **Isolation**: Dependencies don't conflict
- **Portability**: Run anywhere (laptop, cloud, on-prem)
- **Scalability**: Easy horizontal scaling
- **DevOps**: Simplifies CI/CD pipelines

## Core Concepts

### 1. Dockerfile
Defines how to build your container image:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### 2. Docker Image
Built artifact from Dockerfile - immutable template.

### 3. Docker Container
Running instance of an image - your application.

### 4. Docker Compose
Orchestrates multiple containers as one system.

### 5. Volumes
Persist data between container restarts.

## Typical AI Application Stack

```
┌─────────────────┐
│   FastAPI App   │ :8000
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    │         │        │
┌───▼──┐  ┌──▼───┐  ┌─▼────┐
│PostgreSQL Redis │  │ Celery│
│(pgvector)  │     │ Worker │
└──────┘  └──────┘  └───────┘
```

## Docker Commands

### Build & Run
```bash
# Build image
docker build -t my-ai-app:latest .

# Run container
docker run -p 8000:8000 --env-file .env my-ai-app

# Run in background (detached)
docker run -d -p 8000:8000 my-ai-app

# Run interactively
docker run -it my-ai-app /bin/bash
```

### Manage Containers
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# View logs
docker logs <container-id>
docker logs -f <container-id>  # follow

# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# Remove image
docker rmi my-ai-app:latest
```

## Docker Compose Commands

### Start & Stop
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild and start
docker-compose up -d --build

# View logs
docker-compose logs -f app

# Stop and remove volumes
docker-compose down -v
```

### Manage Services
```bash
# List services
docker-compose ps

# Execute command in service
docker-compose exec app python manage.py migrate

# Scale service
docker-compose up -d --scale celery_worker=3

# Restart specific service
docker-compose restart app
```

## Prerequisites

Install Docker:

**Mac**:
```bash
brew install --cask docker
# Or download Docker Desktop
```

**Linux**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

**Windows**:
Download Docker Desktop from docker.com

Install Docker Compose (usually included with Docker Desktop):
```bash
docker-compose --version
```

## Running the Examples

1. **Build and run single container**:
```bash
docker build -t my-ai-app .
docker run -p 8000:8000 --env-file .env my-ai-app
```

2. **Run full stack**:
```bash
# Create .env file first
cp .env.example .env
# Edit .env with your values

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

3. **Access services**:
- App: http://localhost:8000
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- ChromaDB: http://localhost:8001

## Best Practices

### Dockerfile Optimization
1. **Layer caching**: Copy requirements before code
2. **Multi-stage builds**: Separate build and runtime
3. **Slim images**: Use `-slim` or `-alpine` variants
4. **.dockerignore**: Exclude unnecessary files
5. **Non-root user**: Run as non-root for security

### Security
1. **No secrets in images**: Use env vars
2. **Scan images**: `docker scan my-ai-app`
3. **Update base images**: Regularly rebuild
4. **Minimal images**: Less code = fewer vulnerabilities
5. **Health checks**: Monitor container health

### Performance
1. **Resource limits**: Set memory/CPU limits
2. **Volume mounts**: For large data
3. **Build cache**: Use BuildKit
4. **Multi-stage**: Smaller final images
5. **Parallel builds**: Build services concurrently

## Production Considerations

### Environment Variables
```python
# Use pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### Healthchecks
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Logging
```python
import logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Monitoring
- Use Docker stats: `docker stats`
- Prometheus metrics exporter
- Container orchestration (Kubernetes, ECS)

## Common Issues & Solutions

### Issue: Out of Memory
**Solution**: Set memory limits
```yaml
services:
  app:
    mem_limit: 2g
    mem_reservation: 1g
```

### Issue: Slow Builds
**Solution**: Optimize Dockerfile, use BuildKit
```bash
DOCKER_BUILDKIT=1 docker build -t my-app .
```

### Issue: Large Images
**Solution**: Multi-stage builds, slim base images
```dockerfile
FROM python:3.11-slim  # vs python:3.11
```

### Issue: Services Can't Connect
**Solution**: Use service names as hostnames
```python
# Inside container
DATABASE_URL = "postgresql://user:pass@postgres:5432/db"
# Not localhost!
```

## Book References

- `AI_eng.10` - Deployment and production considerations

## Next Steps

After mastering Docker:
- Module 4.2: PostgreSQL + pgvector
- Module 4.3: Observability with Langfuse
- Module 4.5: Async & Background Jobs (Celery in containers)
- Module 4.7: Cloud Deployment (ECS, Kubernetes)
