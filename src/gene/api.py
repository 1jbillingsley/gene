"""FastAPI application for the gene project."""

from fastapi import FastAPI

from .models import Message

app = FastAPI()

__all__ = ["app"]


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns a simple status dictionary indicating the API is running.
    """
    return {"status": "ok"}


@app.post("/messages")
async def create_message(message: Message) -> dict[str, str]:
    """Accept a message and echo its body.

    Validates the request payload using :class:`Message`. This endpoint is a
    placeholder for future AI-driven processing and can be extended with
    routing logic or tool integrations.
    """
    return {"body": message.body}
