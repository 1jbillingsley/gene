"""Tests for configuration validation."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from gene.config import Settings


class TestSettings:
    """Test cases for Settings configuration validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Clear environment variables to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.log_level == "INFO"
            assert settings.openai_api_key is None
            assert settings.openai_model == "gpt-4"
            assert settings.agent_id is None
            assert settings.conversation_memory is True
            assert settings.structured_output is True
            assert settings.max_tokens == 1000
            assert settings.temperature == 0.7

    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        # Test with environment variables set
        with patch.dict(
            os.environ,
            {
                "LOG_LEVEL": "DEBUG",
                "OPENAI_API_KEY": "sk-test123",
                "OPENAI_MODEL": "gpt-4",
                "OPENAI_AGENT_ID": "test-agent-123",
                "OPENAI_CONVERSATION_MEMORY": "false",
                "OPENAI_STRUCTURED_OUTPUT": "false",
                "OPENAI_MAX_TOKENS": "2000",
                "OPENAI_TEMPERATURE": "0.5",
            },
        ):
            settings = Settings()
            assert settings.log_level == "DEBUG"
            assert settings.openai_api_key == "sk-test123"
            assert settings.openai_model == "gpt-4"
            assert settings.agent_id == "test-agent-123"
            assert settings.conversation_memory is False
            assert settings.structured_output is False
            assert settings.max_tokens == 2000
            assert settings.temperature == 0.5

    def test_validate_openai_config_success(self):
        """Test successful OpenAI configuration validation."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_api_key_123"

        # Should not raise any exception
        settings.validate_openai_config()

    def test_validate_openai_config_missing_key(self):
        """Test validation failure when API key is missing."""
        settings = Settings()
        settings.openai_api_key = None

        with pytest.raises(
            ValueError, match="OpenAI Agents SDK API key not configured"
        ):
            settings.validate_openai_config()

    def test_validate_openai_config_empty_key(self):
        """Test validation failure when API key is empty."""
        settings = Settings()
        settings.openai_api_key = "   "  # Whitespace only

        with pytest.raises(ValueError, match="OpenAI Agents SDK API key is empty"):
            settings.validate_openai_config()

    def test_validate_openai_config_invalid_format(self):
        """Test validation failure when API key has invalid format."""
        settings = Settings()
        settings.openai_api_key = "invalid_key_format"

        with pytest.raises(
            ValueError, match="Invalid OpenAI Agents SDK API key format"
        ):
            settings.validate_openai_config()

    def test_validate_openai_config_various_invalid_formats(self):
        """Test validation with various invalid key formats."""
        settings = Settings()

        invalid_keys = [
            "",
            "pk-wrong_prefix",
            "just_a_string",
            "123456789",
            "sk",  # Too short
        ]

        for invalid_key in invalid_keys:
            settings.openai_api_key = invalid_key
            with pytest.raises(ValueError):
                settings.validate_openai_config()

    def test_openai_model_default_when_not_set(self):
        """Test that openai_model defaults to gpt-4 when not set."""
        # Test the default value directly
        with patch("os.getenv") as mock_getenv:

            def getenv_side_effect(key, default=None):
                if key == "OPENAI_MODEL":
                    return default  # Return the default value
                elif key == "LOG_LEVEL":
                    return "INFO"
                elif key == "OPENAI_API_KEY":
                    return None
                return default

            mock_getenv.side_effect = getenv_side_effect
            settings = Settings()
            assert settings.openai_model == "gpt-4"


class TestMainStartup:
    """Test cases for application startup validation."""

    def test_main_startup_with_valid_config(self, capsys):
        """Test that main() starts successfully with valid configuration."""
        from unittest.mock import patch
        from gene.__main__ import main

        # Mock uvicorn.run to prevent actual server startup
        with (
            patch("gene.__main__.uvicorn.run") as mock_run,
            patch("gene.__main__.settings") as mock_settings,
        ):

            mock_settings.validate_openai_config.return_value = None
            mock_settings.log_level = "INFO"

            main()

            # Verify that uvicorn.run was called (meaning validation passed)
            mock_run.assert_called_once()

    def test_main_startup_with_invalid_config(self, capsys):
        """Test that main() fails gracefully with invalid configuration."""
        from unittest.mock import patch
        from gene.__main__ import main

        with patch("gene.__main__.settings") as mock_settings:
            mock_settings.validate_openai_config.side_effect = ValueError(
                "OpenAI Agents SDK API key not configured"
            )

            main()

            # Check that error message was printed
            captured = capsys.readouterr()
            assert (
                "Configuration Error: OpenAI Agents SDK API key not configured"
                in captured.out
            )
            assert (
                "Please check your environment variables and try again." in captured.out
            )


class TestAgentsSDKConfiguration:
    """Test cases for OpenAI Agents SDK specific configuration."""

    def test_agent_configuration_defaults(self):
        """Test that Agents SDK configuration has appropriate defaults."""
        settings = Settings()

        # Test default values for Agents SDK settings
        assert settings.agent_id is None  # Optional, no default
        assert settings.conversation_memory is True
        assert settings.structured_output is True
        assert settings.max_tokens == 1000
        assert settings.temperature == 0.7

    def test_boolean_environment_variable_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", False),  # Only "true" (case-insensitive) should be True
            ("0", False),
            ("", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"OPENAI_CONVERSATION_MEMORY": env_value}):
                settings = Settings()
                assert (
                    settings.conversation_memory == expected
                ), f"Failed for '{env_value}'"

    def test_numeric_environment_variable_parsing(self):
        """Test that numeric environment variables are parsed correctly."""
        with patch.dict(
            os.environ, {"OPENAI_MAX_TOKENS": "2000", "OPENAI_TEMPERATURE": "0.9"}
        ):
            settings = Settings()
            assert settings.max_tokens == 2000
            assert settings.temperature == 0.9

    def test_validate_max_tokens_positive(self):
        """Test validation of max_tokens parameter."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"

        # Valid values should pass
        settings.max_tokens = 1000
        settings.validate_openai_config()  # Should not raise

        settings.max_tokens = 1
        settings.validate_openai_config()  # Should not raise

    def test_validate_max_tokens_invalid(self):
        """Test validation failure for invalid max_tokens values."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"

        invalid_values = [0, -1, -100]
        for invalid_value in invalid_values:
            settings.max_tokens = invalid_value
            with pytest.raises(
                ValueError, match=f"Invalid max_tokens value: {invalid_value}"
            ):
                settings.validate_openai_config()

    def test_validate_temperature_valid_range(self):
        """Test validation of temperature parameter within valid range."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"

        # Valid values should pass
        valid_temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]
        for temp in valid_temperatures:
            settings.temperature = temp
            settings.validate_openai_config()  # Should not raise

    def test_validate_temperature_invalid_range(self):
        """Test validation failure for temperature values outside valid range."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"

        invalid_temperatures = [-0.1, -1.0, 2.1, 3.0, 10.0]
        for temp in invalid_temperatures:
            settings.temperature = temp
            with pytest.raises(ValueError, match=f"Invalid temperature value: {temp}"):
                settings.validate_openai_config()

    def test_validate_model_known_models(self):
        """Test validation with known OpenAI models."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"

        known_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]

        for model in known_models:
            settings.openai_model = model
            settings.validate_openai_config()  # Should not raise

    def test_validate_model_unknown_model_warning(self, caplog):
        """Test that unknown models generate a warning but don't fail validation."""
        import logging

        settings = Settings()
        settings.openai_api_key = "sk-valid_key"
        settings.openai_model = "gpt-5-future-model"

        with caplog.at_level(logging.WARNING):
            settings.validate_openai_config()  # Should not raise

        # Check that a warning was logged
        assert len(caplog.records) == 1
        assert "gpt-5-future-model" in caplog.records[0].message
        assert "not in the list of known models" in caplog.records[0].message

    def test_agent_id_optional(self):
        """Test that agent_id is optional and can be None."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_key"
        settings.agent_id = None

        settings.validate_openai_config()  # Should not raise

        # Also test with a value
        settings.agent_id = "test-agent-123"
        settings.validate_openai_config()  # Should not raise

    def test_comprehensive_agents_sdk_validation(self):
        """Test comprehensive validation with all Agents SDK parameters."""
        settings = Settings()
        settings.openai_api_key = "sk-valid_agents_sdk_key"
        settings.openai_model = "gpt-4"
        settings.agent_id = "production-agent-v1"
        settings.conversation_memory = True
        settings.structured_output = True
        settings.max_tokens = 1500
        settings.temperature = 0.8

        # Should validate successfully with all parameters set
        settings.validate_openai_config()  # Should not raise
