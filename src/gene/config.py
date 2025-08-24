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

    log_level: str = ""
    openai_api_key: str | None = None
    openai_model: str = ""
    
    # OpenAI Agents SDK specific settings
    agent_id: str | None = None
    conversation_memory: bool = True
    structured_output: bool = True
    max_tokens: int = 1000
    temperature: float = 0.7
    
    def __post_init__(self):
        """Load values from environment variables after initialization."""
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.agent_id = os.getenv("OPENAI_AGENT_ID")
        self.conversation_memory = os.getenv("OPENAI_CONVERSATION_MEMORY", "true").lower() == "true"
        self.structured_output = os.getenv("OPENAI_STRUCTURED_OUTPUT", "true").lower() == "true"
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    def validate_openai_config(self) -> None:
        """Validate OpenAI configuration and raise helpful errors.
        
        Raises:
            ValueError: If OpenAI API key is missing or invalid, or if Agents SDK
                       configuration is invalid.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI Agents SDK API key not configured. Please set the OPENAI_API_KEY "
                "environment variable or add it to your .env file."
            )
        
        if not self.openai_api_key.strip():
            raise ValueError(
                "OpenAI Agents SDK API key is empty. Please provide a valid API key."
            )
        
        # Basic format validation - OpenAI keys typically start with 'sk-'
        if not self.openai_api_key.startswith("sk-"):
            raise ValueError(
                "Invalid OpenAI Agents SDK API key format. API keys should start with 'sk-'."
            )
        
        # Validate OpenAI Assistants API (Agents SDK) configuration parameters
        if self.max_tokens <= 0:
            raise ValueError(
                f"Invalid max_tokens value: {self.max_tokens}. Must be a positive integer."
            )
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"Invalid temperature value: {self.temperature}. Must be between 0.0 and 2.0."
            )
        
        # Validate model name (basic check for common OpenAI models)
        valid_models = {
            "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", 
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        }
        if self.openai_model not in valid_models:
            # Log warning but don't fail - allow for newer models
            import logging
            logging.warning(
                f"Model '{self.openai_model}' is not in the list of known models. "
                f"Known models: {', '.join(sorted(valid_models))}"
            )


settings = Settings()

# Configure root logging according to the resolved settings.
logging.basicConfig(level=settings.log_level)
