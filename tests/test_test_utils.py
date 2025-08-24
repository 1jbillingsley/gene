"""Tests for the test utilities module."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import set_client
from gene.openai_client import OpenAIException

from test_utils import (
    MockAgentsSDKClient,
    MockOpenAIClient,
    ErrorMockOpenAIClient,
    OpenAIErrorFactory,
    ClientInjector,
    create_mock_client,
    create_mock_agents_sdk_client,
    inject_mock_client,
    inject_mock_agents_sdk_client,
    reset_client,
)


class TestMockAgentsSDKClient:
    """Test the MockAgentsSDKClient utility."""

    def test_default_behavior(self):
        """Test mock Agents SDK client with default configuration."""
        client = MockAgentsSDKClient()

        result = client.complete("test prompt")

        assert result == "Mock agent response"
        assert client.call_count == 1
        assert client.last_prompt == "test prompt"
        assert client.call_history == ["test prompt"]
        assert client._agent_id == "mock_agent_123"
        assert client._conversation_context["conversation_id"] == "mock_conv_456"
        assert client._conversation_context["turn_count"] == 1

    def test_custom_agent_metadata(self):
        """Test mock client with custom agent metadata."""
        client = MockAgentsSDKClient(
            default_response="Custom agent response",
            agent_id="custom_agent_789",
            conversation_id="custom_conv_012",
        )

        result = client.complete("test")

        assert result == "Custom agent response"
        assert client._agent_id == "custom_agent_789"
        assert client._conversation_context["conversation_id"] == "custom_conv_012"

    def test_conversation_context_tracking(self):
        """Test conversation context management."""
        client = MockAgentsSDKClient()

        # Initial state
        context = client.get_conversation_context()
        assert context["turn_count"] == 0

        # After messages
        client.complete("first")
        client.complete("second")

        context = client.get_conversation_context()
        assert context["turn_count"] == 2
        assert context["last_prompt"] == "second"

    def test_tool_suggestion_simulation(self):
        """Test tool suggestion and execution simulation."""
        client = MockAgentsSDKClient()

        # Test echo tool suggestion
        result = client.complete("suggest echo hello")
        assert "echo tool" in result.lower()
        assert "Tool execution result: hello" in result
        assert client.tool_suggestions == ["echo"]

        # Test reverse tool suggestion
        client.reset()
        result = client.complete("suggest reverse world")
        assert "reverse tool" in result.lower()
        assert "Tool execution result: dlrow" in result
        assert client.tool_suggestions == ["reverse"]


class TestMockOpenAIClient:
    """Test the MockOpenAIClient utility."""

    def test_default_behavior(self):
        """Test mock client with default configuration."""
        client = MockOpenAIClient()

        result = client.complete("test prompt")

        assert result == "Mock AI response"
        assert client.call_count == 1
        assert client.last_prompt == "test prompt"
        assert client.call_history == ["test prompt"]

    def test_custom_default_response(self):
        """Test mock client with custom default response."""
        client = MockOpenAIClient(default_response="Custom response")

        result = client.complete("test")

        assert result == "Custom response"
        assert client.call_count == 1

    def test_specific_responses(self):
        """Test mock client with specific prompt-response mappings."""
        responses = {"hello": "Hi there!", "goodbye": "See you later!"}
        client = MockOpenAIClient(responses=responses)

        # Test specific responses
        assert client.complete("hello") == "Hi there!"
        assert client.complete("goodbye") == "See you later!"

        # Test default for unmapped prompt
        assert client.complete("unknown") == "Mock AI response"

        assert client.call_count == 3

    def test_exception_raising(self):
        """Test mock client that raises exceptions."""
        exception = ValueError("Test error")
        client = MockOpenAIClient(should_raise=exception)

        with pytest.raises(ValueError, match="Test error"):
            client.complete("test")

        assert client.call_count == 1
        assert client.call_history == ["test"]

    def test_call_tracking(self):
        """Test call tracking functionality."""
        client = MockOpenAIClient()

        prompts = ["first", "second", "third"]
        for prompt in prompts:
            client.complete(prompt)

        assert client.call_count == 3
        assert client.call_history == prompts
        assert client.last_prompt == "third"

    def test_reset_functionality(self):
        """Test reset functionality."""
        client = MockOpenAIClient()

        # Make some calls
        client.complete("test1")
        client.complete("test2")

        assert client.call_count == 2
        assert len(client.call_history) == 2

        # Reset
        client.reset()

        assert client.call_count == 0
        assert len(client.call_history) == 0
        assert client.last_prompt is None

    def test_configure_response(self):
        """Test dynamic response configuration."""
        client = MockOpenAIClient()

        # Configure a specific response
        client.configure_response("special", "Special response")

        assert client.complete("special") == "Special response"
        assert client.complete("other") == "Mock AI response"

    def test_configure_exception(self):
        """Test dynamic exception configuration."""
        client = MockOpenAIClient()

        # Initially works normally
        assert client.complete("test") == "Mock AI response"

        # Configure to raise exception
        exception = RuntimeError("Configured error")
        client.configure_exception(exception)

        with pytest.raises(RuntimeError, match="Configured error"):
            client.complete("test")


class TestErrorMockOpenAIClient:
    """Test the ErrorMockOpenAIClient utility."""

    def test_always_raises_exception(self):
        """Test that error mock always raises the configured exception."""
        exception = ValueError("Always fails")
        client = ErrorMockOpenAIClient(exception)

        with pytest.raises(ValueError, match="Always fails"):
            client.complete("test1")

        with pytest.raises(ValueError, match="Always fails"):
            client.complete("test2")

        assert client.call_count == 2
        assert client.call_history == ["test1", "test2"]


class TestOpenAIErrorFactory:
    """Test the OpenAIErrorFactory utility."""

    def test_rate_limit_error(self):
        """Test rate limit error creation."""
        error = OpenAIErrorFactory.rate_limit_error(retry_after=120)

        assert isinstance(error, OpenAIException)
        assert error.error.status_code == 429
        assert error.error.retry_after == 120
        assert error.error.error_type == "rate_limit"
        assert "rate limit exceeded" in error.error.message.lower()

    def test_rate_limit_error_default_retry(self):
        """Test rate limit error with default retry time."""
        error = OpenAIErrorFactory.rate_limit_error()

        assert error.error.retry_after == 60  # Default

    def test_authentication_error(self):
        """Test authentication error creation."""
        error = OpenAIErrorFactory.authentication_error()

        assert isinstance(error, OpenAIException)
        assert error.error.status_code == 401
        assert error.error.error_type == "authentication"
        assert "Invalid OpenAI API credentials" == error.error.message

    def test_connection_error(self):
        """Test connection error creation."""
        error = OpenAIErrorFactory.connection_error()

        assert isinstance(error, OpenAIException)
        assert error.error.status_code == 502
        assert error.error.error_type == "connection"
        assert "Unable to connect to OpenAI service" in error.error.message

    def test_timeout_error(self):
        """Test timeout error creation."""
        error = OpenAIErrorFactory.timeout_error()

        assert isinstance(error, OpenAIException)
        assert error.error.status_code == 504
        assert error.error.error_type == "timeout"
        assert "OpenAI request timed out" in error.error.message

    def test_api_error(self):
        """Test generic API error creation."""
        error = OpenAIErrorFactory.api_error("Custom message")

        assert isinstance(error, OpenAIException)
        assert error.error.status_code == 502
        assert error.error.error_type == "api_error"
        assert "OpenAI API error: Custom message" == error.error.message

    def test_api_error_default_message(self):
        """Test API error with default message."""
        error = OpenAIErrorFactory.api_error()

        assert "OpenAI API error: OpenAI API error" == error.error.message


class TestClientInjector:
    """Test the ClientInjector context manager."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    def test_context_manager_injection(self):
        """Test client injection and restoration with context manager."""
        mock_client = MockOpenAIClient("Injected response")

        # Verify no client initially
        from gene.agent import _client

        assert _client is None

        # Use context manager
        with ClientInjector(mock_client) as injected:
            from gene.agent import _client

            assert _client is mock_client
            assert injected is mock_client

        # Verify restoration
        from gene.agent import _client

        assert _client is None

    def test_context_manager_with_existing_client(self):
        """Test context manager when there's already a client set."""
        original_client = MockOpenAIClient("Original")
        new_client = MockOpenAIClient("New")

        # Set original client
        set_client(original_client)

        # Use context manager with new client
        with ClientInjector(new_client):
            from gene.agent import _client

            assert _client is new_client

        # Verify original is restored
        from gene.agent import _client

        assert _client is original_client

    def test_context_manager_exception_handling(self):
        """Test that context manager restores client even if exception occurs."""
        original_client = MockOpenAIClient("Original")
        new_client = MockOpenAIClient("New")

        set_client(original_client)

        try:
            with ClientInjector(new_client):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify original is still restored
        from gene.agent import _client

        assert _client is original_client


class TestConvenienceFunctions:
    """Test convenience functions for creating and injecting clients."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    def test_create_mock_client_defaults(self):
        """Test create_mock_client with default parameters."""
        client = create_mock_client()

        assert isinstance(client, MockOpenAIClient)
        assert client.complete("test") == "Mock AI response"

    def test_create_mock_client_custom_response(self):
        """Test create_mock_client with custom response."""
        client = create_mock_client(response="Custom response")

        assert client.complete("test") == "Custom response"

    def test_create_mock_client_with_responses(self):
        """Test create_mock_client with specific responses."""
        responses = {"hello": "Hi!"}
        client = create_mock_client(responses=responses)

        assert client.complete("hello") == "Hi!"
        assert client.complete("other") == "Mock AI response"

    def test_create_mock_client_with_exception(self):
        """Test create_mock_client that raises exception."""
        exception = ValueError("Test error")
        client = create_mock_client(exception=exception)

        with pytest.raises(ValueError, match="Test error"):
            client.complete("test")

    def test_inject_mock_client(self):
        """Test inject_mock_client convenience function."""
        client = inject_mock_client("Injected response")

        # Verify client is injected
        from gene.agent import _client

        assert _client is client

        # Verify it works
        assert client.complete("test") == "Injected response"

    def test_create_mock_agents_sdk_client_defaults(self):
        """Test create_mock_agents_sdk_client with default parameters."""
        client = create_mock_agents_sdk_client()

        assert isinstance(client, MockAgentsSDKClient)
        assert client.complete("test") == "Mock agent response"
        assert client._agent_id == "mock_agent_123"
        assert client._conversation_context["conversation_id"] == "mock_conv_456"

    def test_create_mock_agents_sdk_client_custom(self):
        """Test create_mock_agents_sdk_client with custom parameters."""
        client = create_mock_agents_sdk_client(
            response="Custom agent response",
            agent_id="custom_agent",
            conversation_id="custom_conv",
        )

        assert client.complete("test") == "Custom agent response"
        assert client._agent_id == "custom_agent"
        assert client._conversation_context["conversation_id"] == "custom_conv"

    def test_inject_mock_agents_sdk_client(self):
        """Test inject_mock_agents_sdk_client convenience function."""
        client = inject_mock_agents_sdk_client(
            response="Injected agent response", agent_id="injected_agent"
        )

        # Verify client is injected
        from gene.agent import _client

        assert _client is client

        # Verify it works
        assert client.complete("test") == "Injected agent response"
        assert client._agent_id == "injected_agent"

    def test_reset_client(self):
        """Test reset_client convenience function."""
        # Set a client
        mock_client = MockOpenAIClient()
        set_client(mock_client)

        # Verify it's set
        from gene.agent import _client

        assert _client is mock_client

        # Reset
        reset_client()

        # Verify it's cleared
        from gene.agent import _client

        assert _client is None
