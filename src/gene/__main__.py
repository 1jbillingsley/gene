"""Command line interface to run the Gene API server."""

import uvicorn

from .api import app
from .config import settings


def main() -> None:
    """Run the Uvicorn server hosting the API."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
