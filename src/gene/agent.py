"""Core agent interface for message processing."""

from __future__ import annotations

import logging
from typing import Protocol

from .models import ActionResult, Message
from .tools import get_tool

logger = logging.getLogger(__name__)


class OpenAIClient(Protocol):
    """Minimal protocol for an OpenAI-like client.

    This protocol defines the subset of methods the agent expects from an
    OpenAI client. A real implementation can be provided at runtime via
    :func:`set_client`.
    """

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        """Generate a completion for the given prompt."""
        ...


_client: OpenAIClient | None = None


def get_conversation_context() -> dict:
    """Get the current conversation context from the Agents SDK client.
    
    Returns:
        Dictionary containing conversation context, or empty dict if not available.
    """
    from .openai_client import AgentsSDKWrapper
    
    if _client is None:
        return {}
    
    # Check if client has conversation context (either real AgentsSDKWrapper or mock)
    if (isinstance(_client, AgentsSDKWrapper) or 
        hasattr(_client, '_conversation_context')):
        return getattr(_client, '_conversation_context', {})
    
    return {}


def update_conversation_context(context: dict) -> None:
    """Update the conversation context in the Agents SDK client.
    
    Args:
        context: Dictionary containing conversation context to merge
    """
    from .openai_client import AgentsSDKWrapper
    
    if _client is None:
        logger.warning("Cannot update conversation context: Agents SDK client not available")
        return
    
    # Check if client has conversation context (either real AgentsSDKWrapper or mock)
    if not (isinstance(_client, AgentsSDKWrapper) or 
            hasattr(_client, '_conversation_context')):
        logger.warning("Cannot update conversation context: Agents SDK client not available")
        return
    
    if not isinstance(context, dict):
        logger.warning("Cannot update conversation context: context must be a dictionary")
        return
    
    current_context = getattr(_client, '_conversation_context', {})
    current_context.update(context)
    _client._conversation_context = current_context
    logger.debug(f"Updated conversation context with {len(context)} items")


def clear_conversation_context() -> None:
    """Clear the conversation context in the Agents SDK client."""
    from .openai_client import AgentsSDKWrapper
    
    if _client is None:
        logger.warning("Cannot clear conversation context: Agents SDK client not available")
        return
    
    # Check if client has conversation context (either real AgentsSDKWrapper or mock)
    if not (isinstance(_client, AgentsSDKWrapper) or 
            hasattr(_client, '_conversation_context')):
        logger.warning("Cannot clear conversation context: Agents SDK client not available")
        return
    
    _client._conversation_context = {}
    logger.info("Cleared conversation context")


def set_client(client: OpenAIClient | None) -> None:
    """Inject the OpenAI client used for message processing.

    ``client`` may be ``None`` to reset the agent to placeholder mode.
    """
    global _client
    _client = client


def process_message(message: Message) -> ActionResult:
    """Process a message and return the agent's response.

    If a tool is able to handle the message, it is invoked first. Otherwise,
    if no client has been configured, a canned placeholder response is
    returned. When a real client is available, the message body is forwarded to
    it and its output is wrapped in an :class:`ActionResult`.
    
    The returned ActionResult includes enhanced metadata for Agents SDK responses,
    including agent_id, conversation_id, structured_output flags, and processing
    time tracking.
    
    Raises:
        Exception: Propagates any errors from the OpenAI client for proper
                  HTTP error handling at the API layer.
    """
    import time
    from .config import settings
    from .openai_client import AgentsSDKWrapper, OpenAIException
    
    start_time = time.time()
    
    tool = get_tool(message.body)
    if tool is not None:
        logger.info(f"Processing message with tool: {tool.name}")
        processing_time = time.time() - start_time
        return ActionResult(
            reply=tool.handle(message.body), 
            metadata={
                "source": "tool",
                "tool": tool.name,
                "processing_time": processing_time
            }
        )

    if _client is None:
        logger.info("Processing message with placeholder response")
        reply = "This is a placeholder response."
        processing_time = time.time() - start_time
        return ActionResult(
            reply=reply,
            metadata={
                "source": "placeholder",
                "processing_time": processing_time
            }
        )
    else:  # pragma: no cover - depends on external service
        logger.info("Processing message with Agents SDK")
        try:
            reply = _client.complete(message.body)
            processing_time = time.time() - start_time
            
            # Build enhanced metadata for Agents SDK responses
            metadata = {
                "source": "agents_sdk",
                "model": settings.openai_model,
                "processing_time": processing_time,
                "structured_output": True  # Agents SDK uses structured output by default
            }
            
            # Add agent-specific metadata if available (for AgentsSDKWrapper or mock)
            if (isinstance(_client, AgentsSDKWrapper) or 
                hasattr(_client, '_agent_id')):
                # Extract agent_id if available
                agent_id = getattr(_client, '_agent_id', None)
                if agent_id:
                    metadata["agent_id"] = agent_id
                
                # Extract conversation context information
                conversation_context = getattr(_client, '_conversation_context', {})
                if conversation_context:
                    # Add conversation_id if available in context
                    conversation_id = conversation_context.get('conversation_id')
                    if conversation_id:
                        metadata["conversation_id"] = conversation_id
                    
                    # Add context size for debugging/monitoring
                    metadata["conversation_context_size"] = len(conversation_context)
                
                # Add token usage if available from the agent
                if hasattr(_client, '_agent') and _client._agent:
                    agent = _client._agent
                    # Try to get token usage from the last response
                    if hasattr(agent, 'last_response_tokens'):
                        metadata["tokens_used"] = agent.last_response_tokens
                    elif hasattr(agent, 'usage'):
                        usage = getattr(agent, 'usage', {})
                        if isinstance(usage, dict) and 'total_tokens' in usage:
                            metadata["tokens_used"] = usage['total_tokens']
            
            return ActionResult(
                reply=reply,
                metadata=metadata
            )
        except OpenAIException as e:
            # Enhanced error propagation with Agents SDK error information
            logger.error(f"Agents SDK error during message processing: {e.error.message}")
            # Re-raise with the structured error for proper HTTP error handling at API layer
            raise
        except Exception as e:
            # Log the error but re-raise it for proper HTTP error handling at API layer
            logger.error(f"Unexpected error during message processing: {e}")
            raise
