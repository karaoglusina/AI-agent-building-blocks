"""Shared utilities for the learning curriculum."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path for imports
# This allows scripts to import utils regardless of their location
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load environment variables from .env file if it exists
# This allows users to create a .env file in the project root
# Gracefully handle PermissionError (e.g., in sandboxed test environments)
env_path = _project_root / ".env"
if env_path.exists():
    try:
        load_dotenv(env_path)
    except (PermissionError, OSError):
        # .env file exists but can't be read (e.g., sandboxed environment)
        # Scripts should use environment variables directly or mock API calls
        pass

from .data_loader import load_jobs, load_sample_jobs, get_job_by_id
from .models import JobPost, SearchParams
import os

def is_test_mode():
    """Check if running in test mode (TEST_MODE=1)."""
    return os.getenv("TEST_MODE") == "1"

def get_openai_api_key():
    """Get OpenAI API key, returning a dummy key in test mode."""
    if is_test_mode():
        return "sk-test-dummy-key-for-testing"
    return os.getenv("OPENAI_API_KEY")

__all__ = [
    "load_jobs", 
    "load_sample_jobs", 
    "get_job_by_id", 
    "JobPost", 
    "SearchParams",
    "is_test_mode",
    "get_openai_api_key"
]
