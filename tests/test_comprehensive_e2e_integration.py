"""Comprehensive end-to-end integration tests for the complete Gene platform.

This test suite verifies the complete message processing flow from API to Agents SDK,
including conversation context management, tool integration, response format consistency,
configuration loading, and backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, Mock
import os
import tempfile
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from gene.agent import (
    set_client,
    process_message,
    get_conversation_context,
    update_conversation_context,
    clear_conversation_context,
)
from gene.api import app
from gene.config import Settings, load_env_file
from gene.models import Message
from gene.openai_client import (
    get_client,
    AgentsSDKWrapper,
    OpenAIError,
    OpenAIException,
)


class MockAgentsSDKClient:
    """Comprehensive mock Agents SDK client for end-to-end testing."""

    def __init__(
        self,
        agent_id: str = "e2e_agent_123",
        conversation_id: str = "e2e_conv_456",
        model: str = "gpt-4",
    ):
        self.agent_id = agent_id
        self._agent_id = agent_id
        self.model = model
        self._conversation_context = {"conversation_id": conversation_id}
        self._agent = Mock()
        self._agent.last_response_tokens = 200
        self._agent.id = agent_id
        self.call_count = 0
        self.last_prompt = None
        self.conversation_history = []

    def complete(self, prompt: str) -> str:
        """Mock completion that maintains conversation state and tracks usage."""
        self.call_count += 1
        self.last_prompt = prompt
        self.conversation_history.append(prompt)

        # Update conversation context to simulate state management
        self._conversation_context["message_count"] = len(self.conversation_history)
        self._conversation_context["last_message"] = prompt

        # Return contextual response based on conversation history
        if len(self.conversation_history) == 1:
            return f"Hello! I'm processing your first message: {prompt}"
        else:
            return f"Continuing our conversation (message #{len(self.conversation_history)}): {prompt}"


class MockAgentsSDKClientWithErrors:
    """Mock Agents SDK client that can simulate various error conditions."""

    def __init__(self, error_scenario: str = "none"):
        self.error_scenario = error_scenario
        self._agent_id = "error_agent_789"
        self._conversation_context = {"conversation_id": "error_conv_123"}

    def complete(self, prompt: str) -> str:
        """Mock completion that raises specific errors based on scenario."""
        if self.error_scenario == "rate_limit":
            error = OpenAIError(
                message="Agents SDK rate limit exceeded",
                status_code=429,
                retry_after=30,
                error_type="agent_rate_limit",
                agent_id=self._agent_id,
            )
            raise OpenAIException(error)
        elif self.error_scenario == "conversation_error":
            error = OpenAIError(
                message="Conversation context corrupted",
                status_code=502,
                error_type="agent_conversation",
                agent_id=self._agent_id,
                details="Memory overflow detected",
            )
            raise OpenAIException(error)
        elif self.error_scenario == "authentication":
            error = OpenAIError(
                message="Invalid Agents SDK credentials",
                status_code=401,
                error_type="agent_authentication",
                agent_id=self._agent_id,
            )
            raise OpenAIException(error)
        else:
            return f"Mock response for: {prompt}"


class TestComprehensiveEndToEndFlow:
    """Test complete message processing flow from API to Agents SDK."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()

    def test_complete_api_to_agents_sdk_flow(self):
        """Test complete message processing from API endpoint through to Agents SDK."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "Hello, Agents SDK!"})

        # Verify response structure and status
        assert response.status_code == 200
        data = response.json()

        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert (
            data["reply"]
            == "Hello! I'm processing your first message: Hello, Agents SDK!"
        )

        # Verify Agents SDK metadata
        metadata = data["metadata"]
        assert metadata["source"] == "agents_sdk"
        assert metadata["agent_id"] == "e2e_agent_123"
        assert metadata["conversation_id"] == "e2e_conv_456"
        assert metadata["structured_output"] is True
        assert metadata["tokens_used"] == 200
        assert "processing_time" in metadata

        # Verify response headers
        assert response.headers["X-Agent-ID"] == "e2e_agent_123"
        assert response.headers["X-Conversation-ID"] == "e2e_conv_456"
        assert response.headers["X-Processing-Source"] == "agents_sdk"
        assert response.headers["X-Structured-Output"] == "true"

        # Verify client was called correctly
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "Hello, Agents SDK!"

    def test_api_error_propagation_with_agents_sdk_details(self):
        """Test that Agents SDK errors are properly propagated with full context."""
        error_client = MockAgentsSDKClientWithErrors(error_scenario="rate_limit")
        set_client(error_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "Test error handling"})

        # Verify error response structure
        assert response.status_code == 429
        data = response.json()

        assert data["error"] == "Agents SDK rate limit exceeded"
        assert data["error_type"] == "agent_rate_limit"
        assert data["retry_after"] == 30
        assert data["agent_id"] == "error_agent_789"

        # Verify error headers
        assert response.headers["Retry-After"] == "30"
        assert response.headers["X-Agent-ID"] == "error_agent_789"
        assert response.headers["X-Error-Source"] == "agents_sdk"

    def test_conversation_context_persistence_across_multiple_api_calls(self):
        """Test conversation context management across multiple API requests."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)

        # First request
        response1 = client.post("/messages", json={"body": "Start conversation"})
        assert response1.status_code == 200
        data1 = response1.json()

        assert "first message" in data1["reply"]
        assert (
            data1["metadata"]["conversation_context_size"] == 3
        )  # conversation_id + message_count + last_message

        # Second request - should maintain context
        response2 = client.post("/messages", json={"body": "Continue conversation"})
        assert response2.status_code == 200
        data2 = response2.json()

        assert "message #2" in data2["reply"]
        assert (
            data2["metadata"]["conversation_context_size"] == 3
        )  # conversation_id + message_count + last_message
        assert (
            response2.headers["X-Conversation-ID"] == "e2e_conv_456"
        )  # Same conversation

        # Third request - context should continue to grow
        response3 = client.post("/messages", json={"body": "Third message"})
        assert response3.status_code == 200
        data3 = response3.json()

        assert "message #3" in data3["reply"]
        assert (
            data3["metadata"]["conversation_context_size"] == 3
        )  # Still 3 keys, but values updated

        # Verify conversation history was maintained
        assert len(mock_client.conversation_history) == 3
        assert mock_client.conversation_history == [
            "Start conversation",
            "Continue conversation",
            "Third message",
        ]


class TestToolPrecedenceAndIntegration:
    """Test tool precedence and integration with Agents SDK processing."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()

    def test_tool_precedence_over_agents_sdk_echo(self):
        """Test that echo tool takes precedence over Agents SDK processing."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "echo test message"})

        assert response.status_code == 200
        data = response.json()

        # Should be handled by tool, not Agents SDK
        assert data["reply"] == "test message"
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "echo"

        # Agents SDK should not have been called
        assert mock_client.call_count == 0

        # Should not have agent-specific headers
        assert "X-Agent-ID" not in response.headers
        assert "X-Conversation-ID" not in response.headers

    def test_tool_precedence_over_agents_sdk_reverse(self):
        """Test that reverse tool takes precedence over Agents SDK processing."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "reverse hello"})

        assert response.status_code == 200
        data = response.json()

        # Should be handled by tool, not Agents SDK
        assert data["reply"] == "olleh"
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "reverse"

        # Agents SDK should not have been called
        assert mock_client.call_count == 0

    def test_agents_sdk_processing_when_no_tool_matches(self):
        """Test that Agents SDK is used when no tool can handle the message."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "analyze complex data"})

        assert response.status_code == 200
        data = response.json()

        # Should be handled by Agents SDK
        assert "first message" in data["reply"]
        assert data["metadata"]["source"] == "agents_sdk"
        assert data["metadata"]["agent_id"] == "e2e_agent_123"

        # Agents SDK should have been called
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "analyze complex data"

    def test_tool_and_agents_sdk_mixed_conversation(self):
        """Test mixed conversation with both tool usage and Agents SDK processing."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)

        # First: Tool usage
        response1 = client.post("/messages", json={"body": "echo hello"})
        assert response1.status_code == 200
        assert response1.json()["metadata"]["source"] == "tool"
        assert mock_client.call_count == 0  # Agent not called

        # Second: Agents SDK processing
        response2 = client.post("/messages", json={"body": "what did I just say?"})
        assert response2.status_code == 200
        assert response2.json()["metadata"]["source"] == "agents_sdk"
        assert mock_client.call_count == 1  # Agent called once

        # Third: Tool usage again
        response3 = client.post("/messages", json={"body": "reverse world"})
        assert response3.status_code == 200
        assert response3.json()["metadata"]["source"] == "tool"
        assert mock_client.call_count == 1  # Agent still called only once

        # Fourth: Agents SDK processing again
        response4 = client.post("/messages", json={"body": "continue our discussion"})
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["metadata"]["source"] == "agents_sdk"
        assert mock_client.call_count == 2  # Agent called twice

        # Verify conversation context was maintained across tool usage
        assert "message #2" in data4["reply"]  # Should be second agent message


class TestResponseFormatConsistency:
    """Test response format consistency across all processing paths."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()

    def test_response_format_consistency_agents_sdk(self):
        """Test ActionResult format consistency for Agents SDK responses."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "test agents sdk format"})

        assert response.status_code == 200
        data = response.json()

        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)

        # Verify Agents SDK specific metadata
        metadata = data["metadata"]
        required_fields = [
            "source",
            "agent_id",
            "conversation_id",
            "structured_output",
            "processing_time",
            "model",
            "tokens_used",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        assert metadata["source"] == "agents_sdk"
        assert metadata["structured_output"] is True

    def test_response_format_consistency_tools(self):
        """Test ActionResult format consistency for tool responses."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "echo format test"})

        assert response.status_code == 200
        data = response.json()

        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)

        # Verify tool specific metadata
        metadata = data["metadata"]
        required_fields = ["source", "tool", "processing_time"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        assert metadata["source"] == "tool"
        assert metadata["tool"] == "echo"

    def test_response_format_consistency_placeholder(self):
        """Test ActionResult format consistency for placeholder responses."""
        # No client set - should use placeholder
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test placeholder format"})

        assert response.status_code == 200
        data = response.json()

        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)

        # Verify placeholder specific metadata
        metadata = data["metadata"]
        required_fields = ["source", "processing_time"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        assert metadata["source"] == "placeholder"

    def test_error_response_format_consistency(self):
        """Test error response format consistency across different error types."""
        error_client = MockAgentsSDKClientWithErrors(
            error_scenario="conversation_error"
        )
        set_client(error_client)

        client = TestClient(app)
        response = client.post("/messages", json={"body": "test error format"})

        assert response.status_code == 502
        data = response.json()

        # Verify error response structure
        required_fields = ["error", "error_type", "agent_id", "details"]
        for field in required_fields:
            assert field in data, f"Missing required error field: {field}"

        assert data["error_type"] == "agent_conversation"
        assert data["agent_id"] == "error_agent_789"
        assert data["details"] == "Memory overflow detected"


class TestConfigurationAndInitialization:
    """Test configuration loading and agent initialization scenarios."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()
        # Clear cached client
        import gene.openai_client

        gene.openai_client._client = None

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()
        import gene.openai_client

        gene.openai_client._client = None

    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation for Agents SDK."""
        # Test valid configuration
        settings = Settings()
        settings.openai_api_key = "sk-valid123"
        settings.openai_model = "gpt-4"
        settings.validate_openai_config()  # Should not raise

        # Test missing API key
        settings = Settings()
        settings.openai_api_key = None
        with pytest.raises(
            ValueError, match="OpenAI Agents SDK API key not configured"
        ):
            settings.validate_openai_config()

        # Test empty API key
        settings = Settings()
        settings.openai_api_key = "   "
        with pytest.raises(ValueError, match="OpenAI Agents SDK API key is empty"):
            settings.validate_openai_config()

        # Test invalid API key format
        settings = Settings()
        settings.openai_api_key = "invalid-key"
        with pytest.raises(
            ValueError, match="Invalid OpenAI Agents SDK API key format"
        ):
            settings.validate_openai_config()

        # Test invalid max_tokens
        settings = Settings()
        settings.openai_api_key = "sk-valid123"
        settings.max_tokens = -1
        with pytest.raises(ValueError, match="Invalid max_tokens value"):
            settings.validate_openai_config()

        # Test invalid temperature
        settings = Settings()
        settings.openai_api_key = "sk-valid123"
        settings.temperature = 3.0
        with pytest.raises(ValueError, match="Invalid temperature value"):
            settings.validate_openai_config()

    def test_env_file_loading_integration(self):
        """Test complete integration of .env file loading with Agents SDK."""
        # Save original environment variables
        original_env = {}
        env_keys = [
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_AGENT_ID",
            "OPENAI_CONVERSATION_MEMORY",
            "OPENAI_STRUCTURED_OUTPUT",
            "OPENAI_MAX_TOKENS",
            "OPENAI_TEMPERATURE",
        ]

        for key in env_keys:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("OPENAI_API_KEY=sk-env-test-key\n")
            f.write("OPENAI_MODEL=gpt-4-env\n")
            f.write("OPENAI_AGENT_ID=env_agent_123\n")
            f.write("OPENAI_CONVERSATION_MEMORY=false\n")
            f.write("OPENAI_STRUCTURED_OUTPUT=true\n")
            f.write("OPENAI_MAX_TOKENS=2000\n")
            f.write("OPENAI_TEMPERATURE=0.5\n")
            env_file_path = f.name

        try:
            # Load configuration
            load_env_file(Path(env_file_path))

            # Create settings and verify loaded values
            settings = Settings()
            assert settings.openai_api_key == "sk-env-test-key"
            assert settings.openai_model == "gpt-4-env"
            assert settings.agent_id == "env_agent_123"
            assert settings.conversation_memory is False
            assert settings.structured_output is True
            assert settings.max_tokens == 2000
            assert settings.temperature == 0.5

            # Validate configuration
            settings.validate_openai_config()  # Should not raise

        finally:
            # Clean up
            os.unlink(env_file_path)
            # Clean up environment variables
            for key in env_keys:
                if key in os.environ:
                    del os.environ[key]
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value

    @patch("gene.openai_client.Agent")
    def test_agents_sdk_initialization_with_configuration(self, mock_agent_class):
        """Test Agents SDK initialization with proper configuration."""
        # Mock the Agent class
        mock_agent_instance = Mock()
        mock_agent_instance.id = "config_agent_456"
        mock_agent_class.return_value = mock_agent_instance

        # Create wrapper with specific configuration
        wrapper = AgentsSDKWrapper("sk-config-test", "gpt-4")

        # Trigger agent creation
        wrapper.complete("test initialization")

        # Verify agent was created with correct configuration
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]

        expected_config = {
            "api_key": "sk-config-test",
            "model": "gpt-4",
            "conversation_memory": True,
            "structured_output": True,
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        for key, value in expected_config.items():
            assert call_kwargs[key] == value

        # Verify tools were included
        assert "tools" in call_kwargs
        assert isinstance(call_kwargs["tools"], list)

    def test_client_caching_and_reuse(self):
        """Test that Agents SDK client instances are properly cached and reused."""
        with patch("gene.openai_client.Agent") as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_instance.id = "cached_agent_789"
            mock_agent_class.return_value = mock_agent_instance

            with patch("gene.openai_client.settings") as mock_settings:
                mock_settings.openai_api_key = "sk-cache-test"
                mock_settings.openai_model = "gpt-4"

                # First call should create client
                client1 = get_client()

                # Second call should return cached client
                client2 = get_client()

                # Should be the same instance
                assert client1 is client2

                # Agent constructor should only be called once per wrapper
                # (Note: get_client creates wrapper, wrapper creates agent on first use)
                assert isinstance(client1, AgentsSDKWrapper)


class TestBackwardCompatibility:
    """Test backward compatibility with existing API contracts."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()

    def test_existing_api_contract_compatibility(self):
        """Test that existing API contracts remain unchanged."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        client = TestClient(app)

        # Test basic message structure (should remain the same)
        response = client.post("/messages", json={"body": "test compatibility"})

        assert response.status_code == 200
        data = response.json()

        # Core ActionResult structure should be preserved
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)

        # Basic metadata fields should be present
        assert "source" in data["metadata"]
        assert "processing_time" in data["metadata"]

    def test_health_endpoint_unchanged(self):
        """Test that health endpoint remains unchanged."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "ok"}

    def test_validation_error_format_compatibility(self):
        """Test that validation error format remains compatible."""
        client = TestClient(app)
        response = client.post("/messages", json={"body": ""})

        assert response.status_code == 400
        data = response.json()

        # Error format should be consistent
        assert "error" in data
        assert "error_type" in data
        assert data["error_type"] == "validation_error"

    def test_dependency_injection_compatibility(self):
        """Test that dependency injection pattern remains compatible."""
        # Test setting None client
        set_client(None)

        message = Message(body="test dependency injection")
        result = process_message(message)

        assert result.metadata["source"] == "placeholder"

        # Test setting mock client
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        result = process_message(message)
        assert result.metadata["source"] == "agents_sdk"

        # Test resetting to None
        set_client(None)
        result = process_message(message)
        assert result.metadata["source"] == "placeholder"

    def test_tool_interface_compatibility(self):
        """Test that tool interface remains compatible with Agents SDK integration."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # Test that tools still work exactly as before
        message = Message(body="echo backward compatibility")
        result = process_message(message)

        assert result.reply == "backward compatibility"
        assert result.metadata["source"] == "tool"
        assert result.metadata["tool"] == "echo"

        # Verify tool precedence is maintained
        assert mock_client.call_count == 0  # Agent should not be called


class TestConversationContextManagement:
    """Test conversation context management functionality."""

    def setup_method(self):
        """Reset state before each test."""
        set_client(None)
        clear_conversation_context()

    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        clear_conversation_context()

    def test_conversation_context_api_functions(self):
        """Test conversation context management API functions."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # Test getting empty context
        context = get_conversation_context()
        assert context == {"conversation_id": "e2e_conv_456"}

        # Test updating context
        update_conversation_context(
            {"user_preference": "verbose", "session_id": "test_123"}
        )

        updated_context = get_conversation_context()
        assert updated_context["user_preference"] == "verbose"
        assert updated_context["session_id"] == "test_123"
        assert (
            updated_context["conversation_id"] == "e2e_conv_456"
        )  # Original preserved

        # Test clearing context
        clear_conversation_context()
        cleared_context = get_conversation_context()
        assert cleared_context == {}

    def test_conversation_context_persistence_during_processing(self):
        """Test that conversation context persists during message processing."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # Set initial context
        update_conversation_context(
            {"user_id": "test_user", "preferences": {"lang": "en"}}
        )

        # Process message
        message = Message(body="test context persistence")
        result = process_message(message)

        # Verify context was maintained and enhanced
        context = get_conversation_context()
        assert context["user_id"] == "test_user"
        assert context["preferences"]["lang"] == "en"
        assert context["message_count"] == 1  # Added by mock client
        assert context["last_message"] == "test context persistence"

    def test_conversation_context_with_no_client(self):
        """Test conversation context functions when no client is set."""
        # No client set
        context = get_conversation_context()
        assert context == {}

        # Should handle gracefully
        update_conversation_context({"test": "value"})
        context = get_conversation_context()
        assert context == {}

        clear_conversation_context()  # Should not raise error


if __name__ == "__main__":
    pytest.main([__file__])
