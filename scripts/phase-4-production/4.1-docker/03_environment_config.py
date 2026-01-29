"""
03 - Environment Configuration
===============================
Handle environment variables in containers.

Key concept: Never hardcode secrets - use environment variables for configuration in containers.

Book reference: AI_eng.10
"""

import utils._load_env  # Loads .env file automatically

import os
from pydantic_settings import BaseSettings
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


class Settings(BaseSettings):
    """Application settings from environment."""

    # OpenAI
    openai_api_key: str

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/aidb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Langfuse
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None

    # Application
    env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_settings() -> Settings:
    """Load settings from environment."""
    return Settings()


def demonstrate_config():
    """Demonstrate configuration management."""
    print("=== ENVIRONMENT CONFIGURATION ===\n")

    print("1. Create .env file:\n")

    env_content = """OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://user:password@postgres:5432/aidb
REDIS_URL=redis://redis:6379/0
ENV=production
DEBUG=false
LOG_LEVEL=INFO
"""

    print(env_content)

    print("\n2. Load settings in application:\n")

    code = """from config import Settings

settings = Settings()  # Automatically loads from .env

# Use settings
client = OpenAI(api_key=settings.openai_api_key)
db = connect(settings.database_url)
"""

    print(code)

    print("\n3. Override in docker-compose.yml:\n")

    compose_snippet = """services:
  app:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:password@postgres:5432/aidb
      - ENV=production
"""

    print(compose_snippet)


def security_best_practices():
    """Security best practices for env vars."""
    print("\n" + "=" * 70)
    print("=== SECURITY BEST PRACTICES ===\n")

    practices = [
        "1. Never commit .env files to git",
        "   - Add .env to .gitignore",
        "",
        "2. Use .env.example for documentation",
        "   - Shows required vars without secrets",
        "",
        "3. Different .env per environment",
        "   - .env.development",
        "   - .env.staging",
        "   - .env.production",
        "",
        "4. Use secrets management in production",
        "   - AWS Secrets Manager",
        "   - HashiCorp Vault",
        "   - Docker secrets",
        "",
        "5. Validate required variables",
        "   - Use Pydantic for validation",
        "   - Fail fast if missing",
        "",
        "6. Rotate secrets regularly",
        "   - API keys",
        "   - Database passwords",
        "   - Access tokens"
    ]

    for practice in practices:
        print(practice)


def env_example_file():
    """Show .env.example template."""
    print("\n" + "=" * 70)
    print("=== .env.example ===\n")

    example = """# Copy this file to .env and fill in your values

# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/aidb

# Redis
REDIS_URL=redis://redis:6379/0

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=pk-your-key-here
LANGFUSE_SECRET_KEY=sk-your-secret-here

# Application
ENV=development
DEBUG=true
LOG_LEVEL=INFO
"""

    print(example)


if __name__ == "__main__":
    demonstrate_config()
    security_best_practices()
    env_example_file()

    print("\n" + "=" * 70)
    print("\nKey insight: Environment variables = secure, flexible configuration")
    print("\nNever hardcode secrets! Always use env vars in containers.")
