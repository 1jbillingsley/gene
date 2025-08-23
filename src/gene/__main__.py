"""Command line interface to run the Gene API server."""

import uvicorn

from .api import app


def main() -> None:
    """Run the Uvicorn server hosting the API."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
