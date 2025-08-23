"""FastAPI application for the gene project."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request

from .config import settings
from .models import Message

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)

app = FastAPI()

__all__ = ["app"]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log basic information about incoming requests and outgoing responses."""
    logger.info("Request %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response %s %s", response.status_code, request.url.path)
    return response


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
