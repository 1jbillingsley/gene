"""Test utilities for dependency injection and mocking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from gene.agent import OpenAIClient, set_client
from gene.openai_client import OpenAIError, OpenAIException


class MockAgentsSDKClient:
    """Comprehensive mock Agents SDK client for testing.

    This mock client simulates the AgentsSDKWrapper behavior including
    conversation context management, agent metadata, and tool integration.
    """

    def __init__(
        self,
        default_response: str = "Mock agent response",
        responses: Optional[Dict[str, str]] = None,
        should_raise: Optional[Exception] = None,
        agent_id: str = "mock_agent_123",
        conversation_id: str = "mock_conv_456",
    ):
        """Initialize the mock Agents SDK client.

        Args:
            default_response: Default response for any prompt
            responses: Dict mapping specific prompts to specific responses
            should_raise: Exception to raise on complete() calls
            agent_id: Mock agent identifier
            conversation_id: Mock conversation identifier
        """
        self.default_response = default_response
        self.responses = responses or {}
        self.should_raise = should_raise

        # Agent SDK specific attributes
        self._agent_id = agent_id
        self._conversation_context = {
            "conversation_id": conversation_id,
            "turn_count": 0,
            "created_at": "2024-01-01T00:00:00Z",
        }

        # Mock agent with token usage tracking
        self._agent = Mock()
        self._agent.last_response_tokens = 150
        self._agent.usage = {
            "total_tokens": 150,
            "prompt_tokens": 50,
            "completion_tokens": 100,
        }

        # Call tracking
        self.call_count = 0
        self.call_history: List[str] = []
        self.last_prompt: Optional[str] = None

        # Tool integration tracking
        self.tool_suggestions = []
        self.tool_executions = []

    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt

        Returns:
            The configured response for this prompt

        Raises:
            Exception: If should_raise was configured
        """
        self.call_count += 1
        self.last_prompt = prompt
        self.call_history.append(prompt)

        # Update conversation context
        self._conversation_context["turn_count"] += 1
        self._conversation_context["last_prompt"] = prompt

        if self.should_raise:
            raise self.should_raise

        # Handle tool suggestion scenarios
        response = self._handle_tool_suggestions(prompt)
        if response:
            return response

        # Return specific response if configured, otherwise default
        return self.responses.get(prompt, self.default_response)

    def _handle_tool_suggestions(self, prompt: str) -> Optional[str]:
        """Handle tool suggestion scenarios for testing."""
        prompt_lower = prompt.lower()

        # Simulate agent suggesting echo tool
        if "suggest echo" in prompt_lower:
            self.tool_suggestions.append("echo")
            tool_input = prompt[
                prompt.lower().find("suggest echo") + len("suggest echo") :
            ].strip()
            if tool_input:
                # Simulate tool execution within agent context
                tool_result = tool_input  # Echo behavior
                self.tool_executions.append(
                    {"tool": "echo", "input": tool_input, "output": tool_result}
                )
                self._conversation_context["last_tool_used"] = {
                    "tool_name": "echo",
                    "input": tool_input,
                    "output": tool_result,
                }
                return f"I can help you echo that text using the echo tool.\n\nTool execution result: {tool_result}"

        # Simulate agent suggesting reverse tool
        elif "suggest reverse" in prompt_lower:
            self.tool_suggestions.append("reverse")
            tool_input = prompt[
                prompt.lower().find("suggest reverse") + len("suggest reverse") :
            ].strip()
            if tool_input:
                # Simulate tool execution within agent context
                tool_result = tool_input[::-1]  # Reverse behavior
                self.tool_executions.append(
                    {"tool": "reverse", "input": tool_input, "output": tool_result}
                )
                self._conversation_context["last_tool_used"] = {
                    "tool_name": "reverse",
                    "input": tool_input,
                    "output": tool_result,
                }
                return f"I can help you reverse that text using the reverse tool.\n\nTool execution result: {tool_result}"

        return None

    def reset(self) -> None:
        """Reset call tracking and conversation state."""
        self.call_count = 0
        self.call_history.clear()
        self.last_prompt = None
        self.tool_suggestions.clear()
        self.tool_executions.clear()

        # Reset conversation context but keep agent_id
        conversation_id = self._conversation_context.get(
            "conversation_id", "mock_conv_456"
        )
        self._conversation_context = {
            "conversation_id": conversation_id,
            "turn_count": 0,
            "created_at": "2024-01-01T00:00:00Z",
        }

    def configure_response(self, prompt: str, response: str) -> None:
        """Configure a specific response for a prompt."""
        self.responses[prompt] = response

    def configure_exception(self, exception: Exception) -> None:
        """Configure an exception to be raised on complete() calls."""
        self.should_raise = exception

    def update_conversation_context(self, context: Dict[str, Any]) -> None:
        """Update the conversation context."""
        if isinstance(context, dict):
            self._conversation_context.update(context)

    def clear_conversation_context(self) -> None:
        """Clear the conversation context."""
        # Note: This method is for internal mock client use
        # The agent.py clear_conversation_context function directly sets _conversation_context = {}
        agent_id = self._agent_id
        conversation_id = self._conversation_context.get(
            "conversation_id", "mock_conv_456"
        )
        self._conversation_context = {
            "conversation_id": conversation_id,
            "turn_count": 0,
            "created_at": "2024-01-01T00:00:00Z",
        }

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the current conversation context."""
        return self._conversation_context.copy()


class MockOpenAIClient:
    """Comprehensive mock OpenAI client for testing.

    This mock client provides configurable responses and tracks call history
    for verification in tests. It implements the OpenAIClient protocol.
    """

    def __init__(
        self,
        default_response: str = "Mock AI response",
        responses: Optional[Dict[str, str]] = None,
        should_raise: Optional[Exception] = None,
    ):
        """Initialize the mock client.

        Args:
            default_response: Default response for any prompt
            responses: Dict mapping specific prompts to specific responses
            should_raise: Exception to raise on complete() calls
        """
        self.default_response = default_response
        self.responses = responses or {}
        self.should_raise = should_raise

        # Call tracking
        self.call_count = 0
        self.call_history: List[str] = []
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt

        Returns:
            The configured response for this prompt

        Raises:
            Exception: If should_raise was configured
        """
        self.call_count += 1
        self.last_prompt = prompt
        self.call_history.append(prompt)

        if self.should_raise:
            raise self.should_raise

        # Return specific response if configured, otherwise default
        return self.responses.get(prompt, self.default_response)

    def reset(self) -> None:
        """Reset call tracking state."""
        self.call_count = 0
        self.call_history.clear()
        self.last_prompt = None

    def configure_response(self, prompt: str, response: str) -> None:
        """Configure a specific response for a prompt."""
        self.responses[prompt] = response

    def configure_exception(self, exception: Exception) -> None:
        """Configure an exception to be raised on complete() calls."""
        self.should_raise = exception


class ErrorMockOpenAIClient:
    """Mock OpenAI client that always raises exceptions.

    Useful for testing error handling scenarios.
    """

    def __init__(self, exception: Exception):
        """Initialize with the exception to raise.

        Args:
            exception: Exception to raise on complete() calls
        """
        self.exception = exception
        self.call_count = 0
        self.call_history: List[str] = []

    def complete(self, prompt: str) -> str:
        """Always raises the configured exception."""
        self.call_count += 1
        self.call_history.append(prompt)
        raise self.exception


class OpenAIErrorFactory:
    """Factory for creating OpenAI error scenarios."""

    @staticmethod
    def rate_limit_error(retry_after: int = 60) -> OpenAIException:
        """Create a rate limit error."""
        error = OpenAIError(
            message="OpenAI rate limit exceeded. Please try again later.",
            status_code=429,
            retry_after=retry_after,
            error_type="rate_limit",
        )
        return OpenAIException(error)

    @staticmethod
    def authentication_error() -> OpenAIException:
        """Create an authentication error."""
        error = OpenAIError(
            message="Invalid OpenAI API credentials",
            status_code=401,
            error_type="authentication",
        )
        return OpenAIException(error)

    @staticmethod
    def connection_error() -> OpenAIException:
        """Create a connection error."""
        error = OpenAIError(
            message="Unable to connect to OpenAI service. Please check your internet connection.",
            status_code=502,
            error_type="connection",
        )
        return OpenAIException(error)

    @staticmethod
    def timeout_error() -> OpenAIException:
        """Create a timeout error."""
        error = OpenAIError(
            message="OpenAI request timed out. Please try again.",
            status_code=504,
            error_type="timeout",
        )
        return OpenAIException(error)

    @staticmethod
    def api_error(message: str = "OpenAI API error") -> OpenAIException:
        """Create a generic API error."""
        error = OpenAIError(
            message=f"OpenAI API error: {message}",
            status_code=502,
            error_type="api_error",
        )
        return OpenAIException(error)

    # Agents SDK specific error factory methods
    @staticmethod
    def agent_rate_limit_error(
        retry_after: int = 60, agent_id: str = "agent_123"
    ) -> OpenAIException:
        """Create an Agents SDK rate limit error."""
        error = OpenAIError(
            message="Agents SDK rate limit exceeded. Please try again later.",
            status_code=429,
            retry_after=retry_after,
            error_type="agent_rate_limit",
            agent_id=agent_id,
            details="Rate limit exceeded for agent requests",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_authentication_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK authentication error."""
        error = OpenAIError(
            message="Invalid OpenAI Agents SDK credentials or insufficient permissions",
            status_code=401,
            error_type="agent_authentication",
            agent_id=agent_id,
            details="Authentication failed for Agents SDK",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_connection_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK connection error."""
        error = OpenAIError(
            message="Unable to connect to OpenAI Agents SDK service. Please check your internet connection and try again.",
            status_code=502,
            error_type="agent_connection",
            agent_id=agent_id,
            details="Network connection failed for Agents SDK",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_timeout_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK timeout error."""
        error = OpenAIError(
            message="Agents SDK request timed out. The service may be experiencing high load. Please try again.",
            status_code=504,
            error_type="agent_timeout",
            agent_id=agent_id,
            details="Request timeout for Agents SDK",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_conversation_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK conversation error."""
        error = OpenAIError(
            message="Agent conversation context error. The conversation state may be corrupted. Please start a new conversation.",
            status_code=502,
            error_type="agent_conversation",
            agent_id=agent_id,
            details="Conversation context corruption in Agents SDK",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_processing_failure(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK processing failure error."""
        error = OpenAIError(
            message="Agent processing failed. The AI service may be temporarily unavailable. Please try again.",
            status_code=502,
            error_type="agent_processing_failure",
            agent_id=agent_id,
            details="Agent processing pipeline failure",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_validation_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK validation error."""
        error = OpenAIError(
            message="Invalid request to Agents SDK: malformed input",
            status_code=400,
            error_type="agent_validation_error",
            agent_id=agent_id,
            details="Request validation failed for Agents SDK",
        )
        return OpenAIException(error)

    @staticmethod
    def agent_service_unavailable_error(agent_id: str = "agent_123") -> OpenAIException:
        """Create an Agents SDK service unavailable error."""
        error = OpenAIError(
            message="OpenAI Agents SDK service is temporarily unavailable. Please try again later.",
            status_code=503,
            error_type="agent_service_unavailable",
            agent_id=agent_id,
            details="Agents SDK service maintenance or overload",
        )
        return OpenAIException(error)


class ClientInjector:
    """Utility for managing client injection in tests.

    Provides context manager support for clean setup/teardown.
    """

    def __init__(self, client: Optional[OpenAIClient] = None):
        """Initialize with optional client to inject."""
        self.client = client
        self.original_client = None

    def __enter__(self) -> OpenAIClient:
        """Inject the client and return it."""
        # Store original client state
        from gene.agent import _client

        self.original_client = _client

        # Inject our client
        set_client(self.client)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original client state."""
        set_client(self.original_client)


def create_mock_client(
    response: str = "Mock AI response",
    responses: Optional[Dict[str, str]] = None,
    exception: Optional[Exception] = None,
) -> MockOpenAIClient:
    """Convenience function to create a mock client.

    Args:
        response: Default response text
        responses: Dict of prompt -> response mappings
        exception: Exception to raise instead of returning responses

    Returns:
        Configured MockOpenAIClient instance
    """
    if exception:
        return MockOpenAIClient(should_raise=exception)
    return MockOpenAIClient(default_response=response, responses=responses)


def create_mock_agents_sdk_client(
    response: str = "Mock agent response",
    responses: Optional[Dict[str, str]] = None,
    exception: Optional[Exception] = None,
    agent_id: str = "mock_agent_123",
    conversation_id: str = "mock_conv_456",
) -> MockAgentsSDKClient:
    """Convenience function to create a mock Agents SDK client.

    Args:
        response: Default response text
        responses: Dict of prompt -> response mappings
        exception: Exception to raise instead of returning responses
        agent_id: Mock agent identifier
        conversation_id: Mock conversation identifier

    Returns:
        Configured MockAgentsSDKClient instance
    """
    if exception:
        return MockAgentsSDKClient(
            should_raise=exception, agent_id=agent_id, conversation_id=conversation_id
        )
    return MockAgentsSDKClient(
        default_response=response,
        responses=responses,
        agent_id=agent_id,
        conversation_id=conversation_id,
    )


def inject_mock_client(
    response: str = "Mock AI response",
    responses: Optional[Dict[str, str]] = None,
    exception: Optional[Exception] = None,
) -> MockOpenAIClient:
    """Create and inject a mock client, returning the client for verification.

    Args:
        response: Default response text
        responses: Dict of prompt -> response mappings
        exception: Exception to raise instead of returning responses

    Returns:
        The injected MockOpenAIClient instance
    """
    mock_client = create_mock_client(response, responses, exception)
    set_client(mock_client)
    return mock_client


def inject_mock_agents_sdk_client(
    response: str = "Mock agent response",
    responses: Optional[Dict[str, str]] = None,
    exception: Optional[Exception] = None,
    agent_id: str = "mock_agent_123",
    conversation_id: str = "mock_conv_456",
) -> MockAgentsSDKClient:
    """Create and inject a mock Agents SDK client, returning the client for verification.

    Args:
        response: Default response text
        responses: Dict of prompt -> response mappings
        exception: Exception to raise instead of returning responses
        agent_id: Mock agent identifier
        conversation_id: Mock conversation identifier

    Returns:
        The injected MockAgentsSDKClient instance
    """
    mock_client = create_mock_agents_sdk_client(
        response, responses, exception, agent_id, conversation_id
    )
    set_client(mock_client)
    return mock_client


def reset_client() -> None:
    """Reset the agent to use no client (placeholder mode)."""
    set_client(None)
