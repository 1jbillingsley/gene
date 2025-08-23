"""Application configuration loaded from environment variables or a .env file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Settings", "settings"]


def load_env_file(path: Path) -> None:
    """Load key-value pairs from ``path`` into ``os.environ`` if present."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# Load environment variables from a .env file in the project root.
load_env_file(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class Settings:
    """Application settings."""

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

# Configure root logging according to the resolved settings.
logging.basicConfig(level=settings.log_level)
