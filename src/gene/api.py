"""FastAPI application for the gene project."""

from fastapi import FastAPI

app = FastAPI()

__all__ = ["app"]


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns a simple status dictionary indicating the API is running.
    """
    return {"status": "ok"}
