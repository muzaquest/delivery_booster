"""Global configuration for the project.

Use python-dotenv to load environment variables in later steps.
"""

from typing import Optional
import os


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read environment variable with an optional default."""
    return os.getenv(name, default)