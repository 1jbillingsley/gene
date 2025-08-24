"""Command line interface to run the Gene API server."""

import uvicorn

from .api import app
from .config import settings


def main() -> None:
    """Run the Uvicorn server hosting the API."""
    try:
        # Validate OpenAI configuration before starting the server
        settings.validate_openai_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please check your environment variables and try again.")
        return
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
