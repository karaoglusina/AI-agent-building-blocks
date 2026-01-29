"""Load environment variables from .env file.

This module automatically loads environment variables from a .env file
in the project root when imported. Scripts can import this to ensure
.env files are loaded before initializing API clients.

Usage:
    from utils._load_env import load_env  # Explicit import
    # or
    import utils._load_env  # Side-effect import (loads automatically)
"""

from pathlib import Path
from dotenv import load_dotenv

# Find project root (where .env file should be)
# This file is in utils/, so go up one level
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

if env_file.exists():
    try:
        load_dotenv(env_file)
    except (PermissionError, OSError):
        # .env file exists but can't be read (e.g., sandboxed environment)
        # Scripts should use environment variables directly or mock API calls
        pass

def load_env():
    """Explicitly load environment variables from .env file."""
    if env_file.exists():
        try:
            load_dotenv(env_file)
        except (PermissionError, OSError):
            # .env file exists but can't be read (e.g., sandboxed environment)
            # Scripts should use environment variables directly or mock API calls
            pass
