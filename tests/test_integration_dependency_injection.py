"""Integration tests for dependency injection and full message processing flow."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import process_message, set_client
from gene.models import ActionResult, Message
from gene.openai_client import OpenAIException

from test_utils import (
    MockOpenAIClient,
    ErrorMockOpenAIClient,
    OpenAIErrorFactory,
    ClientInjector,
    inject_mock_client,
    reset_client,
)


class TestDependencyInjectionIntegration:
    """Integration tests for dependency injection with mock clients."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    def test_set_client_with_mock_works_correctly(self):
        """Verify existing set_client() function works correctly with mock clients."""
        mock_client = MockOpenAIClient("Test response from mock")

        # Inject the mock client
        set_client(mock_client)

        # Process a message
        message = Message(body="test prompt")
        result = process_message(message)

        # Verify the mock was called and response is correct
        assert result.reply == "Test response from mock"
        assert result.metadata["source"] == "openai"
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "test prompt"

    def test_mock_client_call_tracking(self):
        """Test that mock client properly tracks calls for verification."""
        mock_client = MockOpenAIClient("Response")
        set_client(mock_client)

        # Process multiple messages
        messages = [
            Message(body="first message"),
            Message(body="second message"),
            Message(body="third message"),
        ]

        for msg in messages:
            process_message(msg)

        # Verify call tracking
        assert mock_client.call_count == 3
        assert mock_client.call_history == [
            "first message",
            "second message",
            "third message",
        ]
        assert mock_client.last_prompt == "third message"

    def test_mock_client_specific_responses(self):
        """Test mock client with specific prompt-response mappings."""
        responses = {
            "analyze error": "This is an error analysis",
            "summarize logs": "Log summary provided",
            "default prompt": "Default analysis",
        }

        mock_client = MockOpenAIClient(
            default_response="Generic response", responses=responses
        )
        set_client(mock_client)

        # Test specific responses
        result1 = process_message(Message(body="analyze error"))
        assert result1.reply == "This is an error analysis"

        result2 = process_message(Message(body="summarize logs"))
        assert result2.reply == "Log summary provided"

        # Test default response for unmapped prompt
        result3 = process_message(Message(body="unknown prompt"))
        assert result3.reply == "Generic response"

    def test_client_injector_context_manager(self):
        """Test ClientInjector context manager for clean setup/teardown."""
        mock_client = MockOpenAIClient("Context manager response")

        # Verify no client initially
        result_before = process_message(Message(body="test"))
        assert result_before.metadata["source"] == "placeholder"

        # Use context manager
        with ClientInjector(mock_client) as injected_client:
            result_during = process_message(Message(body="test"))
            assert result_during.reply == "Context manager response"
            assert result_during.metadata["source"] == "openai"
            assert injected_client.call_count == 1

        # Verify client is restored after context
        result_after = process_message(Message(body="test"))
        assert result_after.metadata["source"] == "placeholder"

    def test_inject_mock_client_convenience_function(self):
        """Test the inject_mock_client convenience function."""
        # Test with default response
        mock_client = inject_mock_client("Convenience response")

        result = process_message(Message(body="test"))
        assert result.reply == "Convenience response"
        assert mock_client.call_count == 1

        # Test with specific responses
        reset_client()
        responses = {"specific": "Specific response"}
        mock_client2 = inject_mock_client(responses=responses)

        result1 = process_message(Message(body="specific"))
        assert result1.reply == "Specific response"

        result2 = process_message(Message(body="other"))
        assert result2.reply == "Mock AI response"  # Default


class TestFullMessageProcessingFlow:
    """Integration tests for complete message processing flow with mocked clients."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    @patch("gene.config.settings")
    def test_complete_flow_with_openai_client(self, mock_settings):
        """Test complete message processing flow from API to OpenAI with mock client."""
        mock_settings.openai_model = "gpt-4"

        mock_client = MockOpenAIClient("Complete flow response")
        set_client(mock_client)

        # Create message with metadata
        message = Message(
            body="Analyze this system failure",
            metadata={"source_system": "monitoring", "severity": "high"},
        )

        # Process message
        result = process_message(message)

        # Verify complete flow
        assert isinstance(result, ActionResult)
        assert result.reply == "Complete flow response"
        assert result.metadata == {"source": "openai", "model": "gpt-4"}

        # Verify mock client was called correctly
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "Analyze this system failure"

    def test_tool_precedence_over_openai_processing(self):
        """Test that tools are called first and OpenAI client is not used when tool matches."""
        mock_client = MockOpenAIClient("Should not be called")
        set_client(mock_client)

        # Use reverse tool (should take precedence)
        message = Message(body="reverse hello world")
        result = process_message(message)

        # Verify tool was used, not OpenAI
        assert result.reply == "dlrow olleh"
        assert result.metadata == {"source": "tool", "tool": "reverse"}

        # Verify OpenAI client was never called
        assert mock_client.call_count == 0

        # Test with echo tool
        message2 = Message(body="echo test message")
        result2 = process_message(message2)

        assert result2.reply == "test message"
        assert result2.metadata == {"source": "tool", "tool": "echo"}
        assert mock_client.call_count == 0

    def test_response_format_consistency_across_processing_paths(self):
        """Test that response format is consistent across all processing paths."""
        mock_client = MockOpenAIClient("AI response")
        set_client(mock_client)

        # Test placeholder response format
        set_client(None)
        placeholder_result = process_message(Message(body="test"))
        assert isinstance(placeholder_result, ActionResult)
        assert isinstance(placeholder_result.reply, str)
        assert isinstance(placeholder_result.metadata, dict)
        assert "source" in placeholder_result.metadata

        # Test tool response format
        tool_result = process_message(Message(body="echo test"))
        assert isinstance(tool_result, ActionResult)
        assert isinstance(tool_result.reply, str)
        assert isinstance(tool_result.metadata, dict)
        assert "source" in tool_result.metadata
        assert "tool" in tool_result.metadata

        # Test OpenAI response format
        set_client(mock_client)
        ai_result = process_message(Message(body="test"))
        assert isinstance(ai_result, ActionResult)
        assert isinstance(ai_result.reply, str)
        assert isinstance(ai_result.metadata, dict)
        assert "source" in ai_result.metadata

    def test_error_propagation_through_full_flow(self):
        """Test that errors from OpenAI client are properly propagated through the full flow."""
        # Test rate limit error
        rate_limit_error = OpenAIErrorFactory.rate_limit_error(retry_after=120)
        error_client = ErrorMockOpenAIClient(rate_limit_error)
        set_client(error_client)

        message = Message(body="test message")

        with pytest.raises(OpenAIException) as exc_info:
            process_message(message)

        assert exc_info.value.error.status_code == 429
        assert exc_info.value.error.retry_after == 120
        assert error_client.call_count == 1

        # Test authentication error
        auth_error = OpenAIErrorFactory.authentication_error()
        error_client2 = ErrorMockOpenAIClient(auth_error)
        set_client(error_client2)

        with pytest.raises(OpenAIException) as exc_info:
            process_message(message)

        assert exc_info.value.error.status_code == 401
        assert exc_info.value.error.error_type == "authentication"

    @patch("gene.config.settings")
    def test_configuration_loading_and_client_initialization_flow(self, mock_settings):
        """Test configuration loading and client initialization in processing flow."""
        mock_settings.openai_model = "gpt-3.5-turbo"
        mock_settings.openai_api_key = "sk-test123"

        mock_client = MockOpenAIClient("Config test response")
        set_client(mock_client)

        # Process message and verify configuration is used
        message = Message(body="test configuration")
        result = process_message(message)

        assert result.reply == "Config test response"
        assert result.metadata["model"] == "gpt-3.5-turbo"
        assert result.metadata["source"] == "openai"

    def test_concurrent_message_processing_with_mock_client(self):
        """Test that mock client handles concurrent message processing correctly."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)

        # Configure different responses for different prompts
        mock_client.configure_response("message1", "Response 1")
        mock_client.configure_response("message2", "Response 2")
        mock_client.configure_response("message3", "Response 3")

        messages = [
            Message(body="message1"),
            Message(body="message2"),
            Message(body="message3"),
            Message(body="unknown"),  # Should get default response
        ]

        results = [process_message(msg) for msg in messages]

        # Verify all responses
        assert results[0].reply == "Response 1"
        assert results[1].reply == "Response 2"
        assert results[2].reply == "Response 3"
        assert results[3].reply == "Mock AI response"  # Default

        # Verify call tracking
        assert mock_client.call_count == 4
        assert mock_client.call_history == [
            "message1",
            "message2",
            "message3",
            "unknown",
        ]

    def test_mock_client_reset_functionality(self):
        """Test that mock client reset functionality works correctly."""
        mock_client = MockOpenAIClient("Test response")
        set_client(mock_client)

        # Make some calls
        process_message(Message(body="call1"))
        process_message(Message(body="call2"))

        assert mock_client.call_count == 2
        assert len(mock_client.call_history) == 2

        # Reset and verify
        mock_client.reset()

        assert mock_client.call_count == 0
        assert len(mock_client.call_history) == 0
        assert mock_client.last_prompt is None

        # Make another call after reset
        process_message(Message(body="after_reset"))

        assert mock_client.call_count == 1
        assert mock_client.call_history == ["after_reset"]
        assert mock_client.last_prompt == "after_reset"


class TestErrorScenarioIntegration:
    """Integration tests for various error scenarios with dependency injection."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    def test_all_openai_error_types_through_full_flow(self):
        """Test all OpenAI error types propagate correctly through the full processing flow."""
        error_scenarios = [
            ("rate_limit", OpenAIErrorFactory.rate_limit_error(90)),
            ("authentication", OpenAIErrorFactory.authentication_error()),
            ("connection", OpenAIErrorFactory.connection_error()),
            ("timeout", OpenAIErrorFactory.timeout_error()),
            ("api_error", OpenAIErrorFactory.api_error("Custom API error")),
        ]

        for error_name, error_exception in error_scenarios:
            error_client = ErrorMockOpenAIClient(error_exception)
            set_client(error_client)

            message = Message(body=f"test {error_name}")

            with pytest.raises(OpenAIException) as exc_info:
                process_message(message)

            # Verify error details are preserved
            assert exc_info.value.error.error_type == error_exception.error.error_type
            assert exc_info.value.error.status_code == error_exception.error.status_code
            assert error_client.call_count == 1

            # Reset for next iteration
            reset_client()

    def test_generic_exception_handling_in_full_flow(self):
        """Test that generic exceptions are properly handled in the full flow."""
        generic_error = RuntimeError("Unexpected network failure")
        error_client = ErrorMockOpenAIClient(generic_error)
        set_client(error_client)

        message = Message(body="test generic error")

        with pytest.raises(RuntimeError) as exc_info:
            process_message(message)

        assert str(exc_info.value) == "Unexpected network failure"
        assert error_client.call_count == 1

    def test_error_isolation_between_tool_and_openai_processing(self):
        """Test that errors in OpenAI client don't affect tool processing."""
        # Set up client that would error
        error_client = ErrorMockOpenAIClient(OpenAIErrorFactory.connection_error())
        set_client(error_client)

        # Tool processing should work fine
        tool_message = Message(body="reverse test")
        tool_result = process_message(tool_message)

        assert tool_result.reply == "tset"
        assert tool_result.metadata["source"] == "tool"
        assert error_client.call_count == 0  # Client never called

        # But OpenAI processing should fail
        ai_message = Message(body="analyze this")

        with pytest.raises(OpenAIException):
            process_message(ai_message)

        assert error_client.call_count == 1  # Now client was called and failed
