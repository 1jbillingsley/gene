"""End-to-end integration tests for the complete message processing flow.

These tests verify the complete flow from API endpoint through agent processing
to OpenAI client, ensuring proper tool precedence, response format consistency,
and configuration handling.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import tempfile

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
import pytest

from gene.agent import set_client, process_message
from gene.api import app
from gene.config import Settings, load_env_file
from gene.models import Message, ActionResult
from gene.openai_client import get_client, _OpenAIWrapper, OpenAIError, OpenAIException


class MockOpenAIClient:
    """Mock OpenAI client for end-to-end testing."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.call_count = 0
        self.last_prompt = None
    
    def complete(self, prompt: str) -> str:
        """Mock completion that tracks calls and returns predictable responses."""
        self.call_count += 1
        self.last_prompt = prompt
        return f"AI response to: {prompt}"


class TestEndToEndMessageProcessing:
    """Test complete message processing flow from API to OpenAI."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_complete_api_to_openai_flow(self):
        """Test complete message processing from API endpoint through to OpenAI client."""
        # Setup mock client
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        # Create test client and make request
        client = TestClient(app)
        response = client.post("/messages", json={"body": "Hello, AI!"})
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert data["reply"] == "AI response to: Hello, AI!"
        assert data["metadata"]["source"] == "openai"
        assert "model" in data["metadata"]
        
        # Verify client was called
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "Hello, AI!"
    
    def test_api_to_openai_with_metadata(self):
        """Test that OpenAI responses include proper metadata."""
        mock_client = MockOpenAIClient(model="gpt-4")
        set_client(mock_client)
        
        with patch('gene.config.settings') as mock_settings:
            mock_settings.openai_model = "gpt-4"
            
            client = TestClient(app)
            response = client.post("/messages", json={"body": "Test message"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify metadata includes model information
            assert data["metadata"]["source"] == "openai"
            assert data["metadata"]["model"] == "gpt-4"
    
    def test_api_to_openai_error_propagation(self):
        """Test that OpenAI errors are properly propagated through the full stack."""
        
        class ErrorClient:
            def complete(self, prompt: str) -> str:
                error = OpenAIError(
                    message="Rate limit exceeded",
                    status_code=429,
                    retry_after=60,
                    error_type="rate_limit"
                )
                raise OpenAIException(error)
        
        set_client(ErrorClient())
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "Test message"})
        
        # Verify error response structure
        assert response.status_code == 429
        assert response.headers.get("retry-after") == "60"
        data = response.json()
        assert data["error"] == "Rate limit exceeded"
        assert data["error_type"] == "rate_limit"
        assert data["retry_after"] == 60


class TestToolPrecedenceOverOpenAI:
    """Test that tools are called before OpenAI processing."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_tool_precedence_over_openai_echo(self):
        """Test that echo tool is called instead of OpenAI for 'echo' messages."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "echo hello world"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify tool was used, not OpenAI
        assert data["reply"] == "hello world"
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "echo"
        
        # Verify OpenAI client was never called
        assert mock_client.call_count == 0
    
    def test_tool_precedence_over_openai_reverse(self):
        """Test that reverse tool is called instead of OpenAI for 'reverse' messages."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "reverse hello"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify tool was used, not OpenAI
        assert data["reply"] == "olleh"
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "reverse"
        
        # Verify OpenAI client was never called
        assert mock_client.call_count == 0
    
    def test_openai_used_when_no_tool_matches(self):
        """Test that Agents SDK is used when no tool can handle the message."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "analyze this data"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify Agents SDK was used since no tool matches
        assert data["reply"] == "AI response to: analyze this data"
        assert data["metadata"]["source"] == "agents_sdk"
        
        # Verify OpenAI client was called
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "analyze this data"
    
    def test_agent_level_tool_precedence(self):
        """Test tool precedence at the agent level (not just API level)."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        # Test directly at agent level
        message = Message(body="echo direct test")
        result = process_message(message)
        
        # Verify tool was used
        assert result.reply == "direct test"
        assert result.metadata["source"] == "tool"
        assert result.metadata["tool"] == "echo"
        
        # Verify OpenAI client was never called
        assert mock_client.call_count == 0


class TestResponseFormatConsistency:
    """Test that response format is consistent across all processing paths."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_placeholder_response_format(self):
        """Test that placeholder responses follow ActionResult format."""
        # No client set - should use placeholder
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["source"] == "placeholder"
    
    def test_openai_response_format(self):
        """Test that OpenAI responses follow ActionResult format."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["source"] == "openai"
        assert "model" in data["metadata"]
    
    def test_tool_response_format(self):
        """Test that tool responses follow ActionResult format."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "echo test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify ActionResult structure
        assert "reply" in data
        assert "metadata" in data
        assert isinstance(data["reply"], str)
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "echo"
    
    def test_error_response_format_consistency(self):
        """Test that error responses have consistent format."""
        
        class ErrorClient:
            def complete(self, prompt: str) -> str:
                error = OpenAIError(
                    message="Test error",
                    status_code=500,
                    error_type="test_error"
                )
                raise OpenAIException(error)
        
        set_client(ErrorClient())
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test"})
        
        assert response.status_code == 500
        data = response.json()
        
        # Verify error response structure
        assert "error" in data
        assert "error_type" in data
        assert isinstance(data["error"], str)
        assert isinstance(data["error_type"], str)
        assert data["error"] == "Test error"
        assert data["error_type"] == "test_error"
    
    def test_validation_error_response_format(self):
        """Test that validation errors have consistent format."""
        client = TestClient(app)
        response = client.post("/messages", json={"body": ""})
        
        assert response.status_code == 400
        data = response.json()
        
        # Verify validation error structure
        assert "error" in data
        assert "error_type" in data
        assert data["error"] == "Message body cannot be empty"
        assert data["error_type"] == "validation_error"


class TestConfigurationLoadingAndClientInitialization:
    """Test configuration loading and client initialization scenarios."""
    
    def setup_method(self):
        """Reset client state and clear any cached clients."""
        set_client(None)
        # Clear the cached client in openai_client module
        import gene.openai_client
        gene.openai_client._client = None
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
        import gene.openai_client
        gene.openai_client._client = None
    
    def test_configuration_loading_from_env_file(self):
        """Test that configuration loading function works correctly with .env files."""
        # Test with a new environment variable that doesn't exist
        test_key = "TEST_CONFIG_VAR"
        
        # Ensure the test variable doesn't exist
        if test_key in os.environ:
            del os.environ[test_key]
        
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(f"{test_key}=test_value_from_file\n")
            f.write("OPENAI_MODEL=gpt-4-from-file\n")  # This might not override existing
            env_file_path = f.name
        
        try:
            # Load the env file
            load_env_file(Path(env_file_path))
            
            # Verify the new variable was set from file
            assert os.environ.get(test_key) == "test_value_from_file"
            
            # Test that the load_env_file function works by checking file existence behavior
            # Create a non-existent file path
            non_existent_path = Path("/tmp/non_existent_env_file.env")
            # This should not raise an error
            load_env_file(non_existent_path)
            
        finally:
            # Clean up
            os.unlink(env_file_path)
            if test_key in os.environ:
                del os.environ[test_key]
    
    def test_configuration_env_vars_take_precedence_over_file(self):
        """Test that existing environment variables take precedence over .env file values."""
        # Use test variables that won't conflict with existing environment
        test_key1 = "TEST_ENV_VAR_1"
        test_key2 = "TEST_ENV_VAR_2"
        
        # Set environment variables first
        os.environ[test_key1] = "env-value-1"
        os.environ[test_key2] = "env-value-2"
        
        # Create temporary .env file with different values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(f"{test_key1}=file-value-1\n")
            f.write(f"{test_key2}=file-value-2\n")
            env_file_path = f.name
        
        try:
            # Load the env file
            load_env_file(Path(env_file_path))
            
            # Verify environment variables kept their original values (not overridden by file)
            assert os.environ.get(test_key1) == "env-value-1"
            assert os.environ.get(test_key2) == "env-value-2"
            
        finally:
            # Clean up
            os.unlink(env_file_path)
            for key in [test_key1, test_key2]:
                if key in os.environ:
                    del os.environ[key]
    
    def test_configuration_validation_missing_api_key(self):
        """Test configuration validation when API key is missing."""
        # Create settings with no API key
        settings = Settings(openai_api_key=None)
        
        # Verify validation fails
        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            settings.validate_openai_config()
    
    def test_configuration_validation_empty_api_key(self):
        """Test configuration validation when API key is empty."""
        settings = Settings(openai_api_key="   ")
        
        with pytest.raises(ValueError, match="OpenAI API key is empty"):
            settings.validate_openai_config()
    
    def test_configuration_validation_invalid_api_key_format(self):
        """Test configuration validation when API key has invalid format."""
        settings = Settings(openai_api_key="invalid-key-format")
        
        with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
            settings.validate_openai_config()
    
    def test_configuration_validation_valid_api_key(self):
        """Test configuration validation with valid API key."""
        settings = Settings(openai_api_key="sk-valid123")
        
        # Should not raise any exception
        settings.validate_openai_config()
    
    @patch('gene.openai_client.OpenAI')
    def test_client_initialization_success(self, mock_openai_class):
        """Test successful client initialization with valid configuration."""
        # Mock the OpenAI class
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance
        
        # Create wrapper with valid config
        wrapper = _OpenAIWrapper("sk-test123", "gpt-3.5-turbo")
        
        # Verify OpenAI client was created with correct parameters
        mock_openai_class.assert_called_once_with(api_key="sk-test123")
        assert wrapper._model == "gpt-3.5-turbo"
    
    def test_client_initialization_missing_openai_package(self):
        """Test client initialization when OpenAI package is not available."""
        with patch('gene.openai_client.OpenAI', None):
            with patch('gene.openai_client.settings') as mock_settings:
                mock_settings.openai_api_key = "sk-test123"
                mock_settings.openai_model = "gpt-3.5-turbo"
                
                # Should raise RuntimeError when OpenAI package is missing
                with pytest.raises(RuntimeError, match="openai package is required"):
                    get_client()
    
    def test_client_caching_behavior(self):
        """Test that client instances are properly cached."""
        with patch('gene.openai_client.OpenAI') as mock_openai_class:
            mock_client_instance = MagicMock()
            mock_openai_class.return_value = mock_client_instance
            
            with patch('gene.openai_client.settings') as mock_settings:
                mock_settings.openai_api_key = "sk-test123"
                mock_settings.openai_model = "gpt-3.5-turbo"
                
                # First call should create client
                client1 = get_client()
                
                # Second call should return cached client
                client2 = get_client()
                
                # Should be the same instance
                assert client1 is client2
                
                # OpenAI constructor should only be called once
                assert mock_openai_class.call_count == 1
    
    def test_client_initialization_with_missing_credentials(self):
        """Test client initialization when credentials are missing."""
        with patch('gene.openai_client.settings') as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-3.5-turbo"
            
            # Should return None when credentials are missing
            client = get_client()
            assert client is None
    
    def test_end_to_end_with_configuration_error(self):
        """Test end-to-end flow when configuration is invalid."""
        # Test with missing API key
        with patch('gene.openai_client.settings') as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-3.5-turbo"
            
            # Reset cached client to force re-initialization
            import gene.openai_client
            gene.openai_client._client = None
            
            # Should fall back to placeholder mode
            client = TestClient(app)
            response = client.post("/messages", json={"body": "test message"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["source"] == "placeholder"
    
    def test_end_to_end_configuration_loading_integration(self):
        """Test complete integration of configuration loading with message processing."""
        # Save original environment variables
        original_env = {}
        for key in ["OPENAI_API_KEY", "OPENAI_MODEL"]:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]
        
        # Create temporary .env file with valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=sk-integration-test\n")
            f.write("OPENAI_MODEL=gpt-4-test\n")
            env_file_path = f.name
        
        try:
            # Load configuration
            load_env_file(Path(env_file_path))
            
            # Set up mock client that uses the loaded configuration
            mock_client = MockOpenAIClient(model="gpt-4-test")
            set_client(mock_client)
            
            # Test message processing with loaded configuration
            with patch('gene.config.settings') as mock_settings:
                mock_settings.openai_model = "gpt-4-test"
                
                client = TestClient(app)
                response = client.post("/messages", json={"body": "integration test"})
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response uses loaded configuration
                assert data["reply"] == "AI response to: integration test"
                assert data["metadata"]["source"] == "openai"
                assert data["metadata"]["model"] == "gpt-4-test"
                
        finally:
            # Clean up
            os.unlink(env_file_path)
            for key in ["OPENAI_API_KEY", "OPENAI_MODEL"]:
                if key in os.environ:
                    del os.environ[key]
            # Restore original environment variables
            for key, value in original_env.items():
                os.environ[key] = value