"""Factory for the OpenAI API client used by the agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import settings
from .agent import OpenAIClient

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
    import openai
except Exception:  # pragma: no cover - importlib and runtime errors
    OpenAI = None  # type: ignore
    openai = None  # type: ignore

try:  # pragma: no cover - optional dependency
    # Note: This is a placeholder for the actual OpenAI Agents SDK
    # The real implementation would import from the actual package
    from openai_agents import Agent
except Exception:  # pragma: no cover - importlib and runtime errors
    Agent = None  # type: ignore

logger = logging.getLogger(__name__)

_client: OpenAIClient | None = None


@dataclass
class OpenAIError:
    """Structured error information from OpenAI API calls."""

    message: str
    status_code: int
    retry_after: Optional[int] = None
    error_type: str = "api_error"
    agent_id: Optional[str] = None
    details: Optional[str] = None


class OpenAIException(Exception):
    """Exception that carries structured error information."""

    def __init__(self, error: OpenAIError):
        self.error = error
        super().__init__(error.message)


class _OpenAIWrapper:
    """Light wrapper implementing :class:`OpenAIClient`."""

    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:  # pragma: no cover - network call
        """Generate a completion for the given prompt using OpenAI chat API.

        Args:
            prompt: The user prompt to send to OpenAI

        Returns:
            The AI-generated response text

        Raises:
            ValueError: If the prompt is empty
            OpenAIException: If OpenAI API call fails with structured error information
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            # Convert prompt to chat messages format
            messages = [{"role": "user", "content": prompt.strip()}]

            # Make the API call using the correct chat completions endpoint
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1000,  # Reasonable default
                temperature=0.7,  # Balanced creativity
            )

            # Validate and extract response content
            return self._validate_response(response)

        except OpenAIException:
            # Re-raise OpenAI exceptions with their structured error info
            raise
        except Exception as e:
            # Handle any other unexpected exceptions
            self._handle_api_error(e)

    def _handle_api_error(self, error: Exception) -> str:
        """Handle different types of OpenAI API errors with structured error information.

        Args:
            error: The exception that occurred during API call

        Raises:
            OpenAIException: With structured error information for proper HTTP response handling
        """
        if openai is None:
            logger.error(f"OpenAI API call failed: {error}")
            error_info = OpenAIError(
                message="OpenAI service is not available",
                status_code=503,
                error_type="service_unavailable",
            )
            raise OpenAIException(error_info) from error

        # Handle specific OpenAI error types by checking class names for better testability
        error_class_name = error.__class__.__name__

        if error_class_name == "RateLimitError":
            logger.warning(f"OpenAI rate limit exceeded: {error}")
            # Extract retry-after from error if available
            retry_after = (
                getattr(error, "retry_after", None) or 60
            )  # Default 60 seconds
            error_info = OpenAIError(
                message="OpenAI rate limit exceeded. Please try again later.",
                status_code=429,
                retry_after=retry_after,
                error_type="rate_limit",
            )
            raise OpenAIException(error_info) from error

        elif error_class_name == "AuthenticationError":
            logger.error(f"OpenAI authentication failed: {error}")
            error_info = OpenAIError(
                message="Invalid OpenAI API credentials",
                status_code=401,
                error_type="authentication",
            )
            raise OpenAIException(error_info) from error

        elif error_class_name == "APIConnectionError":
            logger.error(f"OpenAI connection error: {error}")
            error_info = OpenAIError(
                message="Unable to connect to OpenAI service. Please check your internet connection.",
                status_code=502,
                error_type="connection",
            )
            raise OpenAIException(error_info) from error

        elif error_class_name == "APITimeoutError":
            logger.error(f"OpenAI request timeout: {error}")
            error_info = OpenAIError(
                message="OpenAI request timed out. Please try again.",
                status_code=504,
                error_type="timeout",
            )
            raise OpenAIException(error_info) from error

        elif error_class_name == "BadRequestError":
            logger.error(f"OpenAI bad request: {error}")
            error_info = OpenAIError(
                message="Invalid request to OpenAI API",
                status_code=400,
                error_type="bad_request",
            )
            raise OpenAIException(error_info) from error

        elif error_class_name == "APIError":
            logger.error(f"OpenAI API error: {error}")
            error_info = OpenAIError(
                message=f"OpenAI API error: {error}",
                status_code=502,
                error_type="api_error",
            )
            raise OpenAIException(error_info) from error

        else:
            # Handle unexpected errors
            logger.error(f"Unexpected error during OpenAI API call: {error}")
            error_info = OpenAIError(
                message=f"Unexpected error occurred: {error}",
                status_code=500,
                error_type="internal_error",
            )
            raise OpenAIException(error_info) from error

    def _validate_response(self, response) -> str:
        """Validate OpenAI API response and extract content.

        Args:
            response: The OpenAI API response object

        Returns:
            The extracted response content

        Raises:
            OpenAIException: If response format is invalid or empty
        """
        try:
            if not response or not hasattr(response, "choices"):
                raise ValueError("Invalid response format from OpenAI API")

            if not response.choices:
                raise ValueError("No response choices returned from OpenAI API")

            choice = response.choices[0]
            if not hasattr(choice, "message") or not hasattr(choice.message, "content"):
                raise ValueError("Invalid response structure from OpenAI API")

            content = choice.message.content
            if not content or not content.strip():
                raise ValueError("Empty response content from OpenAI API")

            return content.strip()

        except ValueError as e:
            logger.error(f"Response validation failed: {e}")
            error_info = OpenAIError(
                message=f"Invalid response from OpenAI API: {e}",
                status_code=502,
                error_type="invalid_response",
            )
            raise OpenAIException(error_info) from e


class AgentsSDKWrapper:
    """Wrapper implementing OpenAIClient protocol using OpenAI Agents SDK."""

    def __init__(self, api_key: str, model: str):
        """Initialize the Agents SDK wrapper.

        Args:
            api_key: OpenAI API key for authentication
            model: Model name to use for agent processing
        """
        self._api_key = api_key
        self._model = model
        self._agent = None
        self._conversation_context = {}
        self._agent_id = None
        self._available_tools = None

    def complete(self, prompt: str) -> str:
        """Generate a completion using the OpenAI Agents SDK.

        Args:
            prompt: The user prompt to send to the agent

        Returns:
            The AI-generated response text

        Raises:
            ValueError: If the prompt is empty
            OpenAIException: If Agents SDK call fails with structured error information
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            # Ensure agent is initialized
            if self._agent is None:
                self._agent = self._create_agent()

            # Check if any existing tools can handle this prompt
            # This allows the agent to be aware of available tools
            available_tools = self._get_available_tools()

            # Process message through Agents SDK with tool context
            # The agent can now be aware of available tools and potentially suggest their use
            response = self._agent.process_message(
                message=prompt.strip(),
                context=self._conversation_context,
                available_tools=available_tools,
            )

            # Check if the agent response suggests using a tool
            # This allows the agent to recommend tool usage even when tools weren't directly invoked
            processed_response = self._handle_tool_suggestions(response, prompt.strip())

            # Update conversation context for future messages
            if hasattr(response, "context") and response.context:
                if isinstance(response.context, dict):
                    self._conversation_context.update(response.context)

            # Extract and validate response content
            return self._process_agent_response(processed_response or response)

        except OpenAIException:
            # Re-raise OpenAI exceptions with their structured error info
            raise
        except Exception as e:
            # Handle any other unexpected exceptions from Agents SDK
            self._handle_agent_error(e)

    def _get_available_tools(self) -> list[dict]:
        """Get information about available tools for the Agents SDK.

        Returns:
            List of tool descriptions that the agent can be aware of
        """
        if self._available_tools is not None:
            return self._available_tools

        try:
            # Import tools registry to discover available tools
            from .tools import _REGISTRY

            tools_info = []
            for tool in _REGISTRY:
                tool_info = {
                    "name": tool.name,
                    "description": f"Tool that can handle messages starting with '{tool.name} '",
                    "type": "external_tool",
                    "handler": tool.__class__.__name__,
                }

                # Try to get more detailed description from docstring
                if tool.__class__.__doc__:
                    tool_info["description"] = tool.__class__.__doc__.strip()

                tools_info.append(tool_info)

            self._available_tools = tools_info
            logger.debug(f"Discovered {len(tools_info)} available tools for Agents SDK")
            return tools_info

        except Exception as e:
            logger.warning(f"Failed to discover available tools: {e}")
            self._available_tools = []
            return []

    def _handle_tool_suggestions(self, response, original_prompt: str):
        """Handle tool suggestions from the agent response.

        This method allows the agent to suggest tool usage and potentially
        execute tools within the agent context when appropriate.

        Args:
            response: The agent response that might contain tool suggestions
            original_prompt: The original user prompt

        Returns:
            Modified response if tool execution occurred, None otherwise
        """
        try:
            # Extract response content to check for tool suggestions
            response_text = ""
            if hasattr(response, "content") and response.content:
                response_text = str(response.content)
            elif hasattr(response, "message") and response.message:
                response_text = str(response.message)
            elif isinstance(response, str):
                response_text = response

            # Check if the agent suggests using a specific tool
            # This is a simple heuristic - in a real implementation, the Agents SDK
            # might have more sophisticated tool integration
            if (
                "use echo tool" in response_text.lower()
                or "echo tool can" in response_text.lower()
            ):
                from .tools import get_tool

                # Extract the part after "suggest echo" for tool execution
                if "suggest echo" in original_prompt.lower():
                    tool_input = original_prompt[
                        original_prompt.lower().find("suggest echo")
                        + len("suggest echo") :
                    ].strip()
                    tool = get_tool("echo " + tool_input)
                    if tool:
                        tool_result = tool.handle("echo " + tool_input)
                        # Create enhanced response that includes both agent insight and tool result
                        enhanced_response = (
                            f"{response_text}\n\nTool execution result: {tool_result}"
                        )

                        # Update conversation context with tool usage
                        self._conversation_context["last_tool_used"] = {
                            "tool_name": tool.name,
                            "input": tool_input,
                            "output": tool_result,
                        }

                        # Return modified response object
                        if hasattr(response, "content"):
                            response.content = enhanced_response
                        elif hasattr(response, "message"):
                            response.message = enhanced_response
                        else:
                            response = enhanced_response

                        return response

            elif (
                "use reverse tool" in response_text.lower()
                or "reverse tool can" in response_text.lower()
            ):
                from .tools import get_tool

                # Extract the part after "suggest reverse" for tool execution
                if "suggest reverse" in original_prompt.lower():
                    tool_input = original_prompt[
                        original_prompt.lower().find("suggest reverse")
                        + len("suggest reverse") :
                    ].strip()
                    tool = get_tool("reverse " + tool_input)
                    if tool:
                        tool_result = tool.handle("reverse " + tool_input)
                        # Create enhanced response that includes both agent insight and tool result
                        enhanced_response = (
                            f"{response_text}\n\nTool execution result: {tool_result}"
                        )

                        # Update conversation context with tool usage
                        self._conversation_context["last_tool_used"] = {
                            "tool_name": tool.name,
                            "input": tool_input,
                            "output": tool_result,
                        }

                        # Return modified response object
                        if hasattr(response, "content"):
                            response.content = enhanced_response
                        elif hasattr(response, "message"):
                            response.message = enhanced_response
                        else:
                            response = enhanced_response

                        return response

            # No tool suggestions found, return None to use original response
            return None

        except Exception as e:
            logger.warning(f"Error handling tool suggestions: {e}")
            return None

    def _create_agent(self) -> Agent:
        """Create and configure an OpenAI Agent instance.

        Returns:
            Configured Agent instance

        Raises:
            OpenAIException: If agent creation fails
        """
        if Agent is None:
            error_info = OpenAIError(
                message="OpenAI Agents SDK is not available",
                status_code=503,
                error_type="service_unavailable",
            )
            raise OpenAIException(error_info)

        try:
            # Get available tools for agent configuration
            available_tools = self._get_available_tools()

            # Create agent with conversation management and tool awareness
            agent_config = {
                "api_key": self._api_key,
                "model": self._model,
                "conversation_memory": True,
                "structured_output": True,
                "max_tokens": 1000,
                "temperature": 0.7,
            }

            # Add tools to agent configuration if available
            if available_tools:
                agent_config["tools"] = available_tools

            agent = Agent(**agent_config)

            # Capture agent ID if available
            self._agent_id = getattr(agent, "id", None) or getattr(
                agent, "agent_id", None
            )

            logger.info(
                f"Created OpenAI Agent with model: {self._model}, agent_id: {self._agent_id}, tools: {len(available_tools)}"
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to create OpenAI Agent: {e}")
            error_info = OpenAIError(
                message=f"Failed to initialize OpenAI Agent: {e}",
                status_code=500,
                error_type="agent_initialization",
                details=str(e),
            )
            raise OpenAIException(error_info) from e

    def _process_agent_response(self, response) -> str:
        """Process and validate agent response.

        Args:
            response: The response object from Agents SDK

        Returns:
            The extracted response content

        Raises:
            OpenAIException: If response format is invalid or empty
        """
        try:
            # Handle None response
            if response is None:
                raise ValueError("Empty response content from Agents SDK")

            # Handle different response formats from Agents SDK
            if hasattr(response, "content") and response.content is not None:
                content = response.content
            elif hasattr(response, "message") and response.message is not None:
                content = response.message
            elif isinstance(response, str):
                content = response
            else:
                # Try to extract content from response dict
                if isinstance(response, dict):
                    content = response.get("content") or response.get("message")
                    if content is None:
                        content = str(response) if response else None
                else:
                    # For non-dict responses, only use str() if it's not a Mock or similar object
                    # that would have attributes but no meaningful string representation
                    if hasattr(response, "content") or hasattr(response, "message"):
                        # This is likely a response object with None attributes
                        content = None
                    else:
                        content = str(response)

            if not content or not str(content).strip():
                raise ValueError("Empty response content from Agents SDK")

            return str(content).strip()

        except ValueError as e:
            logger.error(f"Agent response validation failed: {e}")
            error_info = OpenAIError(
                message=f"Invalid response from Agents SDK: {e}",
                status_code=502,
                error_type="invalid_agent_response",
                agent_id=self._agent_id,
                details=str(e),
            )
            raise OpenAIException(error_info) from e

    def _handle_agent_error(self, error: Exception) -> None:
        """Handle different types of Agents SDK errors with comprehensive error mapping.

        Args:
            error: The exception that occurred during agent processing

        Raises:
            OpenAIException: With structured error information for proper HTTP response handling
        """
        error_class_name = error.__class__.__name__
        error_message = str(error).lower()

        # Handle Agents SDK specific rate limit errors
        if (
            "RateLimitError" in error_class_name
            or "rate_limit" in error_message
            or "rate limit" in error_message
            or "quota" in error_message
            or "too many requests" in error_message
        ):
            logger.warning(f"Agents SDK rate limit exceeded: {error}")
            retry_after = getattr(error, "retry_after", None) or 60
            error_info = OpenAIError(
                message="Agents SDK rate limit exceeded. Please try again later.",
                status_code=429,
                retry_after=retry_after,
                error_type="agent_rate_limit",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle authentication and authorization errors
        elif (
            "AuthenticationError" in error_class_name
            or "PermissionError" in error_class_name
            or "authentication" in error_message
            or "unauthorized" in error_message
            or "invalid api key" in error_message
            or "permission denied" in error_message
        ):
            logger.error(f"Agents SDK authentication failed: {error}")
            error_info = OpenAIError(
                message="Invalid OpenAI Agents SDK credentials or insufficient permissions",
                status_code=401,
                error_type="agent_authentication",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle network connection errors
        elif (
            "ConnectionError" in error_class_name
            or "NetworkError" in error_class_name
            or "connection" in error_message
            or "network" in error_message
            or "unreachable" in error_message
            or "connection refused" in error_message
            or "connection reset" in error_message
        ):
            logger.error(f"Agents SDK connection error: {error}")
            error_info = OpenAIError(
                message="Unable to connect to OpenAI Agents SDK service. Please check your internet connection and try again.",
                status_code=502,
                error_type="agent_connection",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle timeout errors
        elif (
            "TimeoutError" in error_class_name
            or "timeout" in error_message
            or "timed out" in error_message
            or "deadline exceeded" in error_message
        ):
            logger.error(f"Agents SDK request timeout: {error}")
            error_info = OpenAIError(
                message="Agents SDK request timed out. The service may be experiencing high load. Please try again.",
                status_code=504,
                error_type="agent_timeout",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle conversation context errors
        elif (
            "ConversationError" in error_class_name
            or "ContextError" in error_class_name
            or "conversation" in error_message
            or "context" in error_message
            or "memory" in error_message
            or "session" in error_message
        ):
            logger.error(f"Agents SDK conversation error: {error}")
            error_info = OpenAIError(
                message="Agent conversation context error. The conversation state may be corrupted. Please start a new conversation.",
                status_code=502,
                error_type="agent_conversation",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle agent processing failures
        elif (
            "AgentError" in error_class_name
            or "ProcessingError" in error_class_name
            or "agent failed" in error_message
            or "processing failed" in error_message
            or "agent unavailable" in error_message
        ):
            logger.error(f"Agents SDK processing failure: {error}")
            error_info = OpenAIError(
                message="Agent processing failed. The AI service may be temporarily unavailable. Please try again.",
                status_code=502,
                error_type="agent_processing_failure",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle validation and bad request errors
        elif (
            "ValidationError" in error_class_name
            or "BadRequestError" in error_class_name
            or "invalid" in error_message
            or "malformed" in error_message
            or "bad request" in error_message
        ):
            logger.error(f"Agents SDK validation error: {error}")
            error_info = OpenAIError(
                message=f"Invalid request to Agents SDK: {error}",
                status_code=400,
                error_type="agent_validation_error",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        # Handle service unavailable errors
        elif (
            "ServiceUnavailableError" in error_class_name
            or "service unavailable" in error_message
            or "temporarily unavailable" in error_message
            or "maintenance" in error_message
        ):
            logger.error(f"Agents SDK service unavailable: {error}")
            error_info = OpenAIError(
                message="OpenAI Agents SDK service is temporarily unavailable. Please try again later.",
                status_code=503,
                error_type="agent_service_unavailable",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error

        else:
            # Handle unexpected errors from Agents SDK
            logger.error(f"Unexpected error during Agents SDK processing: {error}")
            error_info = OpenAIError(
                message="Unexpected agent processing error. Please try again or contact support if the issue persists.",
                status_code=500,
                error_type="agent_internal_error",
                agent_id=self._agent_id,
                details=str(error),
            )
            raise OpenAIException(error_info) from error


def create_client(api_key: str, model: str) -> OpenAIClient:
    """Create a new OpenAI client instance using Agents SDK.

    Args:
        api_key: OpenAI API key for authentication
        model: Model name to use for agent processing

    Returns:
        AgentsSDKWrapper instance implementing OpenAIClient protocol

    Raises:
        RuntimeError: If openai-agents package is not installed
    """
    if Agent is None:
        raise RuntimeError("openai-agents package is required but not installed")

    return AgentsSDKWrapper(api_key, model)


def get_client() -> OpenAIClient | None:
    """Return a cached OpenAI client instance using Agents SDK.

    If the required credentials are missing or the OpenAI Agents SDK package is not
    installed, ``None`` is returned and the agent will fall back to placeholder
    responses.
    """

    global _client
    if _client is not None:
        return _client

    if not settings.openai_api_key or not settings.openai_model:
        logger.warning("OpenAI credentials not configured; running in placeholder mode")
        return None

    if Agent is None:
        logger.warning("OpenAI Agents SDK not available; running in placeholder mode")
        return None

    try:
        _client = create_client(settings.openai_api_key, settings.openai_model)
        return _client
    except Exception as e:
        logger.warning(
            f"Failed to create Agents SDK client: {e}; running in placeholder mode"
        )
        return None
