"""Tests for OpenAI client implementation."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from gene.openai_client import (
    _OpenAIWrapper,
    AgentsSDKWrapper,
    get_client,
    create_client,
    OpenAIError,
    OpenAIException,
)


class TestOpenAIWrapper:
    """Test cases for the OpenAI client wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "sk-test123"
        self.model = "gpt-3.5-turbo"

    @patch("gene.openai_client.OpenAI")
    def test_init_creates_client(self, mock_openai_class):
        """Test that wrapper initializes OpenAI client correctly."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        mock_openai_class.assert_called_once_with(api_key=self.api_key)
        assert wrapper._client == mock_client
        assert wrapper._model == self.model

    @patch("gene.openai_client.OpenAI")
    def test_complete_success(self, mock_openai_class):
        """Test successful completion with valid response."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response from AI"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)
        result = wrapper.complete("Test prompt")

        assert result == "Test response from AI"

        # Verify API call was made correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model=self.model,
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=1000,
            temperature=0.7,
        )

    @patch("gene.openai_client.OpenAI")
    def test_complete_strips_whitespace(self, mock_openai_class):
        """Test that prompt and response whitespace is handled correctly."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "  Response with whitespace  "

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)
        result = wrapper.complete("  Prompt with whitespace  ")

        assert result == "Response with whitespace"

        # Verify prompt was stripped in API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["content"] == "Prompt with whitespace"

    @patch("gene.openai_client.OpenAI")
    def test_complete_empty_prompt_raises_error(self, mock_openai_class):
        """Test that empty prompt raises ValueError."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("   ")

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_rate_limit_error(self, mock_openai_module, mock_openai_class):
        """Test handling of rate limit errors with structured error information."""

        # Create a proper exception class with correct name
        class MockRateLimitError(Exception):
            def __init__(self, message, retry_after=None):
                super().__init__(message)
                self.retry_after = retry_after

        # Set the class name to match what the error handler expects
        MockRateLimitError.__name__ = "RateLimitError"

        mock_openai_module.RateLimitError = MockRateLimitError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockRateLimitError(
            "Rate limit exceeded", retry_after=120
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 429
        assert error.retry_after == 120
        assert error.error_type == "rate_limit"
        assert "rate limit exceeded" in error.message.lower()

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_authentication_error(self, mock_openai_module, mock_openai_class):
        """Test handling of authentication errors with structured error information."""

        class MockAuthenticationError(Exception):
            pass

        MockAuthenticationError.__name__ = "AuthenticationError"

        mock_openai_module.AuthenticationError = MockAuthenticationError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAuthenticationError(
            "Invalid API key"
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 401
        assert error.error_type == "authentication"
        assert "Invalid OpenAI API credentials" == error.message

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_connection_error(self, mock_openai_module, mock_openai_class):
        """Test handling of connection errors with structured error information."""

        class MockAPIConnectionError(Exception):
            pass

        MockAPIConnectionError.__name__ = "APIConnectionError"

        mock_openai_module.APIConnectionError = MockAPIConnectionError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAPIConnectionError(
            "Connection failed"
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "connection"
        assert "Unable to connect to OpenAI service" in error.message

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_timeout_error(self, mock_openai_module, mock_openai_class):
        """Test handling of timeout errors with structured error information."""

        class MockAPITimeoutError(Exception):
            pass

        MockAPITimeoutError.__name__ = "APITimeoutError"

        mock_openai_module.APITimeoutError = MockAPITimeoutError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAPITimeoutError(
            "Request timed out"
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 504
        assert error.error_type == "timeout"
        assert "OpenAI request timed out" in error.message

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_bad_request_error(self, mock_openai_module, mock_openai_class):
        """Test handling of bad request errors with structured error information."""

        class MockBadRequestError(Exception):
            pass

        MockBadRequestError.__name__ = "BadRequestError"

        mock_openai_module.BadRequestError = MockBadRequestError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockBadRequestError(
            "Bad request"
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 400
        assert error.error_type == "bad_request"
        assert "Invalid request to OpenAI API" == error.message

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_complete_generic_api_error(self, mock_openai_module, mock_openai_class):
        """Test handling of generic API errors with structured error information."""

        class MockAPIError(Exception):
            pass

        MockAPIError.__name__ = "APIError"

        mock_openai_module.APIError = MockAPIError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAPIError("API error")
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "api_error"
        assert "OpenAI API error: API error" == error.message

    @patch("gene.openai_client.OpenAI")
    def test_complete_unexpected_error(self, mock_openai_class):
        """Test handling of unexpected errors with structured error information."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Unexpected error")
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 500
        assert error.error_type == "internal_error"
        assert "Unexpected error occurred: Unexpected error" == error.message

    @patch("gene.openai_client.OpenAI")
    def test_validate_response_success(self, mock_openai_class):
        """Test successful response validation."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        # Valid response
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = "Valid response"

        result = wrapper._validate_response(response)
        assert result == "Valid response"

    @patch("gene.openai_client.OpenAI")
    def test_validate_response_no_choices(self, mock_openai_class):
        """Test response validation with no choices."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        response = Mock()
        response.choices = []

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_response"
        assert "No response choices returned" in error.message

    @patch("gene.openai_client.OpenAI")
    def test_validate_response_invalid_structure(self, mock_openai_class):
        """Test response validation with invalid structure."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        # Missing message attribute
        response = Mock()
        response.choices = [Mock()]
        del response.choices[0].message

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_response"
        assert "Invalid response structure" in error.message

    @patch("gene.openai_client.OpenAI")
    def test_validate_response_empty_content(self, mock_openai_class):
        """Test response validation with empty content."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = ""

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_response"
        assert "Empty response content" in error.message

    @patch("gene.openai_client.OpenAI")
    def test_validate_response_none_response(self, mock_openai_class):
        """Test response validation with None response."""
        wrapper = _OpenAIWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(None)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_response"
        assert "Invalid response format" in error.message


class TestGetClient:
    """Test cases for the get_client factory function."""

    def setup_method(self):
        """Reset global client state before each test."""
        import gene.openai_client

        gene.openai_client._client = None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent", None)
    def test_get_client_openai_not_installed(self, mock_settings):
        """Test get_client when Agents SDK is not installed returns None."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-3.5-turbo"

        result = get_client()
        assert result is None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.OpenAI")
    def test_get_client_missing_credentials(self, mock_openai_class, mock_settings):
        """Test get_client with missing credentials returns None."""
        mock_settings.openai_api_key = None
        mock_settings.openai_model = "gpt-3.5-turbo"

        result = get_client()
        assert result is None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.OpenAI")
    def test_get_client_missing_model(self, mock_openai_class, mock_settings):
        """Test get_client with missing model returns None."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = None

        result = get_client()
        assert result is None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent")
    def test_get_client_success(self, mock_agent_class, mock_settings):
        """Test successful client creation with Agents SDK."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-3.5-turbo"

        result = get_client()

        assert result is not None
        assert isinstance(result, AgentsSDKWrapper)

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent")
    def test_get_client_caching(self, mock_agent_class, mock_settings):
        """Test that Agents SDK client is cached and reused."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-3.5-turbo"

        # First call
        result1 = get_client()
        # Second call
        result2 = get_client()

        assert result1 is result2
        assert isinstance(result1, AgentsSDKWrapper)


class TestMockOpenAIClient:
    """Test cases for creating mock clients for testing."""

    def test_mock_client_interface(self):
        """Test that mock client implements the required interface."""

        class MockOpenAIClient:
            def complete(self, prompt: str) -> str:
                return f"Mock response for: {prompt}"

        mock_client = MockOpenAIClient()
        result = mock_client.complete("test prompt")

        assert result == "Mock response for: test prompt"

    def test_mock_client_with_agent(self):
        """Test using mock client with the agent system."""
        from gene.agent import set_client, process_message
        from gene.models import Message

        class MockOpenAIClient:
            def complete(self, prompt: str) -> str:
                return f"Mock AI response: {prompt}"

        # Inject mock client
        set_client(MockOpenAIClient())

        # Test message processing
        message = Message(body="test message")
        result = process_message(message)

        assert result.reply == "Mock AI response: test message"

        # Clean up
        set_client(None)


class TestOpenAIErrorHandling:
    """Test cases for structured error handling."""

    def test_openai_error_creation(self):
        """Test OpenAIError dataclass creation."""
        error = OpenAIError(
            message="Test error",
            status_code=429,
            retry_after=60,
            error_type="rate_limit",
        )

        assert error.message == "Test error"
        assert error.status_code == 429
        assert error.retry_after == 60
        assert error.error_type == "rate_limit"

    def test_openai_error_defaults(self):
        """Test OpenAIError with default values."""
        error = OpenAIError(message="Test error", status_code=500)

        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.retry_after is None
        assert error.error_type == "api_error"

    def test_openai_exception_creation(self):
        """Test OpenAIException creation and behavior."""
        error_info = OpenAIError(
            message="Test error message",
            status_code=429,
            retry_after=120,
            error_type="rate_limit",
        )

        exception = OpenAIException(error_info)

        assert exception.error == error_info
        assert str(exception) == "Test error message"

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai")
    def test_rate_limit_with_default_retry_after(
        self, mock_openai_module, mock_openai_class
    ):
        """Test rate limit error when no retry_after is provided."""

        class MockRateLimitError(Exception):
            def __init__(self, message):
                super().__init__(message)
                # No retry_after attribute

        MockRateLimitError.__name__ = "RateLimitError"
        mock_openai_module.RateLimitError = MockRateLimitError

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockRateLimitError(
            "Rate limit exceeded"
        )
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper("sk-test", "gpt-3.5-turbo")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 429
        assert error.retry_after == 60  # Default value
        assert error.error_type == "rate_limit"

    @patch("gene.openai_client.OpenAI")
    @patch("gene.openai_client.openai", None)
    def test_openai_not_available_error(self, mock_openai_class):
        """Test error handling when OpenAI module is not available."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Some error")
        mock_openai_class.return_value = mock_client

        wrapper = _OpenAIWrapper("sk-test", "gpt-3.5-turbo")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 503
        assert error.error_type == "service_unavailable"
        assert "OpenAI service is not available" == error.message

    @patch("gene.openai_client.OpenAI")
    def test_response_validation_comprehensive(self, mock_openai_class):
        """Test comprehensive response validation scenarios."""
        wrapper = _OpenAIWrapper("sk-test", "gpt-3.5-turbo")

        # Test response without choices attribute
        response_no_choices_attr = Mock(spec=[])  # No choices attribute
        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response_no_choices_attr)
        assert exc_info.value.error.error_type == "invalid_response"

        # Test choice without message
        response_no_message = Mock()
        response_no_message.choices = [Mock(spec=[])]  # No message attribute
        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response_no_message)
        assert exc_info.value.error.error_type == "invalid_response"

        # Test message without content
        response_no_content = Mock()
        response_no_content.choices = [Mock()]
        response_no_content.choices[0].message = Mock(spec=[])  # No content attribute
        with pytest.raises(OpenAIException) as exc_info:
            wrapper._validate_response(response_no_content)
        assert exc_info.value.error.error_type == "invalid_response"


class TestAgentsSDKWrapper:
    """Test cases for the Agents SDK wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "sk-test123"
        self.model = "gpt-4"

    @patch("gene.openai_client.Agent")
    def test_init_creates_wrapper(self, mock_agent_class):
        """Test that wrapper initializes correctly."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        assert wrapper._api_key == self.api_key
        assert wrapper._model == self.model
        assert wrapper._agent is None
        assert wrapper._conversation_context == {}

    @patch("gene.openai_client.Agent")
    def test_complete_success(self, mock_agent_class):
        """Test successful completion with valid agent response."""
        # Setup mock agent and response
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "Test response from agent"
        mock_response.context = {"conversation_id": "test_123"}
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)
        result = wrapper.complete("Test prompt")

        assert result == "Test response from agent"

        # Verify agent was created and called correctly
        mock_agent_class.assert_called_once_with(
            api_key=self.api_key,
            model=self.model,
            conversation_memory=True,
            structured_output=True,
            max_tokens=1000,
            temperature=0.7,
        )

        # Verify process_message was called with correct message
        mock_agent.process_message.assert_called_once()
        call_args = mock_agent.process_message.call_args
        assert call_args[1]["message"] == "Test prompt"
        # Context should have been empty initially (before update)

        # Verify conversation context was updated
        assert wrapper._conversation_context == {"conversation_id": "test_123"}

    @patch("gene.openai_client.Agent")
    def test_complete_with_conversation_context(self, mock_agent_class):
        """Test completion with existing conversation context."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "Follow-up response"
        mock_response.context = {"conversation_id": "test_123", "turn": 2}
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)
        wrapper._conversation_context = {"conversation_id": "test_123", "turn": 1}

        result = wrapper.complete("Follow-up prompt")

        assert result == "Follow-up response"

        # Verify process_message was called with correct message
        mock_agent.process_message.assert_called_once()
        call_args = mock_agent.process_message.call_args
        assert call_args[1]["message"] == "Follow-up prompt"
        # Context should have contained the initial context

        # Verify conversation context was updated
        assert wrapper._conversation_context == {
            "conversation_id": "test_123",
            "turn": 2,
        }

    @patch("gene.openai_client.Agent")
    def test_complete_strips_whitespace(self, mock_agent_class):
        """Test that prompt whitespace is handled correctly."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "  Response with whitespace  "
        # Don't set context to avoid the Mock.keys() issue
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)
        result = wrapper.complete("  Prompt with whitespace  ")

        assert result == "Response with whitespace"

        # Verify prompt was stripped in agent call
        mock_agent.process_message.assert_called_once()
        call_args = mock_agent.process_message.call_args
        assert call_args[1]["message"] == "Prompt with whitespace"

    @patch("gene.openai_client.Agent")
    def test_complete_empty_prompt_raises_error(self, mock_agent_class):
        """Test that empty prompt raises ValueError."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("   ")

    @patch("gene.openai_client.Agent")
    def test_complete_agent_creation_failure(self, mock_agent_class):
        """Test handling of agent creation failure."""
        mock_agent_class.side_effect = Exception("Agent creation failed")

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 500
        assert error.error_type == "agent_initialization"
        assert "Failed to initialize OpenAI Agent" in error.message

    @patch("gene.openai_client.Agent", None)
    def test_complete_agent_sdk_not_available(self):
        """Test handling when Agents SDK is not available."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 503
        assert error.error_type == "service_unavailable"
        assert "OpenAI Agents SDK is not available" == error.message

    @patch("gene.openai_client.Agent")
    def test_complete_agent_rate_limit_error(self, mock_agent_class):
        """Test handling of agent rate limit errors."""

        class MockAgentRateLimitError(Exception):
            def __init__(self, message, retry_after=None):
                super().__init__(message)
                self.retry_after = retry_after

        MockAgentRateLimitError.__name__ = "AgentRateLimitError"

        mock_agent = Mock()
        mock_agent.process_message.side_effect = MockAgentRateLimitError(
            "Agent rate limit exceeded", retry_after=90
        )
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 429
        assert error.retry_after == 90
        assert error.error_type == "agent_rate_limit"
        assert "Agents SDK rate limit exceeded" in error.message

    @patch("gene.openai_client.Agent")
    def test_complete_agent_authentication_error(self, mock_agent_class):
        """Test handling of agent authentication errors."""

        class MockAgentAuthError(Exception):
            pass

        MockAgentAuthError.__name__ = "AgentAuthenticationError"

        mock_agent = Mock()
        mock_agent.process_message.side_effect = MockAgentAuthError(
            "Invalid agent credentials"
        )
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 401
        assert error.error_type == "agent_authentication"
        assert (
            "Invalid OpenAI Agents SDK credentials or insufficient permissions"
            == error.message
        )

    @patch("gene.openai_client.Agent")
    def test_complete_agent_connection_error(self, mock_agent_class):
        """Test handling of agent connection errors."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("connection failed")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "agent_connection"
        assert "Unable to connect to OpenAI Agents SDK service" in error.message

    @patch("gene.openai_client.Agent")
    def test_complete_agent_timeout_error(self, mock_agent_class):
        """Test handling of agent timeout errors."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("timeout occurred")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 504
        assert error.error_type == "agent_timeout"
        assert "Agents SDK request timed out" in error.message

    @patch("gene.openai_client.Agent")
    def test_complete_agent_conversation_error(self, mock_agent_class):
        """Test handling of agent conversation errors."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("conversation context error")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "agent_conversation"
        assert "Agent conversation context error" in error.message

    @patch("gene.openai_client.Agent")
    def test_complete_unexpected_agent_error(self, mock_agent_class):
        """Test handling of unexpected agent errors."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Unexpected agent error")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("Test prompt")

        error = exc_info.value.error
        assert error.status_code == 500
        assert error.error_type == "agent_internal_error"
        assert (
            "Unexpected agent processing error. Please try again or contact support if the issue persists."
            == error.message
        )

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_content_attribute(self, mock_agent_class):
        """Test processing agent response with content attribute."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = Mock()
        response.content = "Response with content attribute"

        result = wrapper._process_agent_response(response)
        assert result == "Response with content attribute"

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_message_attribute(self, mock_agent_class):
        """Test processing agent response with message attribute."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = Mock()
        response.message = "Response with message attribute"
        # Remove content attribute to test fallback
        del response.content

        result = wrapper._process_agent_response(response)
        assert result == "Response with message attribute"

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_string_response(self, mock_agent_class):
        """Test processing string agent response."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        result = wrapper._process_agent_response("Direct string response")
        assert result == "Direct string response"

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_dict_with_content(self, mock_agent_class):
        """Test processing dict agent response with content key."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = {"content": "Dict response with content"}

        result = wrapper._process_agent_response(response)
        assert result == "Dict response with content"

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_dict_with_message(self, mock_agent_class):
        """Test processing dict agent response with message key."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = {"message": "Dict response with message"}

        result = wrapper._process_agent_response(response)
        assert result == "Dict response with message"

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_dict_fallback(self, mock_agent_class):
        """Test processing dict agent response with fallback to string conversion."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = {"other_key": "Some value"}

        result = wrapper._process_agent_response(response)
        assert "other_key" in result and "Some value" in result

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_empty_content(self, mock_agent_class):
        """Test processing agent response with empty content."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = Mock()
        response.content = ""

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._process_agent_response(response)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_agent_response"
        assert "Empty response content from Agents SDK" in error.message

    @patch("gene.openai_client.Agent")
    def test_process_agent_response_none_content(self, mock_agent_class):
        """Test processing agent response with None content and no message."""
        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        response = Mock()
        response.content = None
        response.message = None  # Ensure message is also None

        with pytest.raises(OpenAIException) as exc_info:
            wrapper._process_agent_response(response)

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_agent_response"
        assert "Empty response content from Agents SDK" in error.message

    @patch("gene.openai_client.Agent")
    def test_agent_caching(self, mock_agent_class):
        """Test that agent instance is cached and reused."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "First response"
        # Don't set context to avoid the Mock.keys() issue
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)

        # First call
        result1 = wrapper.complete("First prompt")
        # Second call
        result2 = wrapper.complete("Second prompt")

        assert result1 == "First response"
        assert result2 == "First response"

        # Agent should only be created once due to caching
        mock_agent_class.assert_called_once()
        # But process_message should be called twice
        assert mock_agent.process_message.call_count == 2

    @patch("gene.openai_client.Agent")
    def test_response_without_context(self, mock_agent_class):
        """Test handling response without context attribute."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "Response without context"
        # Explicitly remove context attribute to test the hasattr check
        if hasattr(mock_response, "context"):
            delattr(mock_response, "context")
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper(self.api_key, self.model)
        result = wrapper.complete("Test prompt")

        assert result == "Response without context"
        # Conversation context should remain unchanged
        assert wrapper._conversation_context == {}


class TestCreateClient:
    """Test cases for the create_client factory function."""

    @patch("gene.openai_client.Agent")
    def test_create_client_success(self, mock_agent_class):
        """Test successful client creation with Agents SDK."""
        result = create_client("sk-test123", "gpt-4")

        assert result is not None
        assert isinstance(result, AgentsSDKWrapper)
        assert result._api_key == "sk-test123"
        assert result._model == "gpt-4"

    @patch("gene.openai_client.Agent", None)
    def test_create_client_agents_sdk_not_installed(self):
        """Test create_client when Agents SDK is not installed."""
        with pytest.raises(RuntimeError, match="openai-agents package is required"):
            create_client("sk-test123", "gpt-4")


class TestGetClientWithAgentsSDK:
    """Test cases for get_client with Agents SDK integration."""

    def setup_method(self):
        """Reset global client state before each test."""
        import gene.openai_client

        gene.openai_client._client = None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent", None)
    def test_get_client_agents_sdk_not_installed(self, mock_settings):
        """Test get_client when Agents SDK is not installed returns None."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-4"

        result = get_client()
        assert result is None

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent")
    def test_get_client_success_with_agents_sdk(self, mock_agent_class, mock_settings):
        """Test successful client creation with Agents SDK."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-4"

        result = get_client()

        assert result is not None
        assert isinstance(result, AgentsSDKWrapper)

    @patch("gene.openai_client.settings")
    @patch("gene.openai_client.Agent")
    def test_get_client_caching_with_agents_sdk(self, mock_agent_class, mock_settings):
        """Test that Agents SDK client is cached and reused."""
        mock_settings.openai_api_key = "sk-test123"
        mock_settings.openai_model = "gpt-4"

        # First call
        result1 = get_client()
        # Second call
        result2 = get_client()

        assert result1 is result2
        assert isinstance(result1, AgentsSDKWrapper)


class TestMockAgentsSDKClient:
    """Test cases for creating mock Agents SDK clients for testing."""

    def test_mock_agents_sdk_client_interface(self):
        """Test that mock Agents SDK client implements the required interface."""

        class MockAgentsSDKClient:
            def complete(self, prompt: str) -> str:
                return f"Mock agent response for: {prompt}"

        mock_client = MockAgentsSDKClient()
        result = mock_client.complete("test prompt")

        assert result == "Mock agent response for: test prompt"

    def test_mock_agents_sdk_client_with_agent(self):
        """Test using mock Agents SDK client with the agent system."""
        from gene.agent import set_client, process_message
        from gene.models import Message

        class MockAgentsSDKClient:
            def complete(self, prompt: str) -> str:
                return f"Mock Agents SDK response: {prompt}"

        # Inject mock client
        set_client(MockAgentsSDKClient())

        # Test message processing
        message = Message(body="test message")
        result = process_message(message)

        assert result.reply == "Mock Agents SDK response: test message"

        # Clean up
        set_client(None)
