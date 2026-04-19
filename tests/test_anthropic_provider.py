"""Tests for Anthropic provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.shared import ProviderType


class TestAnthropicProvider:
    """Test Anthropic provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = AnthropicModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.ANTHROPIC
        assert provider._client is None  # Lazy init
        assert provider._base_url is None

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = AnthropicModelProvider("test-key", base_url="https://custom.anthropic.com")
        assert provider.api_key == "test-key"
        assert provider._base_url == "https://custom.anthropic.com"

    def test_model_validation(self):
        """Test model name validation."""
        provider = AnthropicModelProvider("test-key")

        # Test valid models (canonical names)
        assert provider.validate_model_name("claude-opus-4-6") is True
        assert provider.validate_model_name("claude-sonnet-4-6") is True
        assert provider.validate_model_name("claude-haiku-4-5-20251001") is True
        assert provider.validate_model_name("claude-3-5-sonnet-20241022") is True
        assert provider.validate_model_name("claude-3-5-haiku-20241022") is True

        # Test valid aliases
        assert provider.validate_model_name("opus") is True
        assert provider.validate_model_name("sonnet") is True
        assert provider.validate_model_name("haiku") is True
        assert provider.validate_model_name("claude-opus") is True
        assert provider.validate_model_name("claude-sonnet") is True
        assert provider.validate_model_name("claude-haiku") is True

        # Test invalid models
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    def test_resolve_model_name(self):
        """Test model name resolution."""
        provider = AnthropicModelProvider("test-key")

        # Test alias resolution - "opus" and "claude-opus" now point to 4.7
        assert provider._resolve_model_name("opus") == "claude-opus-4-7"
        assert provider._resolve_model_name("sonnet") == "claude-sonnet-4-6"
        assert provider._resolve_model_name("haiku") == "claude-haiku-4-5-20251001"
        assert provider._resolve_model_name("claude-opus") == "claude-opus-4-7"
        assert provider._resolve_model_name("claude-sonnet") == "claude-sonnet-4-6"
        assert provider._resolve_model_name("opus-4.6") == "claude-opus-4-6"

        # Test full name passthrough
        assert provider._resolve_model_name("claude-opus-4-7") == "claude-opus-4-7"
        assert provider._resolve_model_name("claude-opus-4-6") == "claude-opus-4-6"
        assert provider._resolve_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_get_capabilities_opus_4_7(self):
        """Test getting model capabilities for Claude Opus 4.7."""
        provider = AnthropicModelProvider("test-key")

        capabilities = provider.get_capabilities("claude-opus-4-7")
        assert capabilities.model_name == "claude-opus-4-7"
        assert capabilities.friendly_name == "Claude (Opus 4.7)"
        assert capabilities.context_window == 1_000_000
        assert capabilities.max_output_tokens == 128_000
        assert capabilities.max_thinking_tokens == 128_000
        assert capabilities.provider == ProviderType.ANTHROPIC
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_images is True
        assert capabilities.supports_function_calling is True
        assert capabilities.allow_code_generation is True
        assert capabilities.supports_temperature is False

    def test_get_capabilities_opus_4_6(self):
        """Test getting model capabilities for Claude Opus 4.6."""
        provider = AnthropicModelProvider("test-key")

        capabilities = provider.get_capabilities("claude-opus-4-6")
        assert capabilities.model_name == "claude-opus-4-6"
        assert capabilities.context_window == 1_000_000
        assert capabilities.max_output_tokens == 128_000
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_temperature is True

    def test_get_capabilities_sonnet(self):
        """Test getting model capabilities for Claude Sonnet 4.6."""
        provider = AnthropicModelProvider("test-key")

        capabilities = provider.get_capabilities("claude-sonnet-4-6")
        assert capabilities.model_name == "claude-sonnet-4-6"
        # Sourced from LiteLLM which tracks Anthropic's 1M-context window for Sonnet 4.6.
        assert capabilities.context_window == 1_000_000
        assert capabilities.max_output_tokens == 64_000
        assert capabilities.supports_extended_thinking is True
        assert capabilities.allow_code_generation is True

    def test_get_capabilities_haiku(self):
        """Test getting model capabilities for Claude Haiku 4.5."""
        provider = AnthropicModelProvider("test-key")

        capabilities = provider.get_capabilities("claude-haiku-4-5-20251001")
        assert capabilities.model_name == "claude-haiku-4-5-20251001"
        assert capabilities.context_window == 200_000
        assert capabilities.max_output_tokens == 8_192
        assert capabilities.supports_extended_thinking is False
        assert capabilities.allow_code_generation is False

    def test_get_capabilities_with_shorthand(self):
        """Test getting model capabilities with shorthand aliases."""
        provider = AnthropicModelProvider("test-key")

        capabilities = provider.get_capabilities("opus")
        assert capabilities.model_name == "claude-opus-4-7"
        assert capabilities.context_window == 1_000_000

        capabilities = provider.get_capabilities("opus-4.6")
        assert capabilities.model_name == "claude-opus-4-6"

        capabilities = provider.get_capabilities("sonnet")
        assert capabilities.model_name == "claude-sonnet-4-6"

        capabilities = provider.get_capabilities("haiku")
        assert capabilities.model_name == "claude-haiku-4-5-20251001"

    def test_unsupported_model_capabilities(self):
        """Test error handling for unsupported models."""
        provider = AnthropicModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider anthropic"):
            provider.get_capabilities("invalid-model")

    def test_extended_thinking_flags(self):
        """Verify correct models support extended thinking."""
        provider = AnthropicModelProvider("test-key")

        # Models that should support thinking
        assert provider.get_capabilities("claude-opus-4-7").supports_extended_thinking is True
        assert provider.get_capabilities("claude-opus-4-6").supports_extended_thinking is True
        assert provider.get_capabilities("claude-sonnet-4-6").supports_extended_thinking is True
        assert provider.get_capabilities("opus").supports_extended_thinking is True
        assert provider.get_capabilities("sonnet").supports_extended_thinking is True

        # Models that should NOT support thinking
        assert provider.get_capabilities("claude-haiku-4-5-20251001").supports_extended_thinking is False
        assert provider.get_capabilities("claude-3-5-sonnet-20241022").supports_extended_thinking is False
        assert provider.get_capabilities("claude-3-5-haiku-20241022").supports_extended_thinking is False

    def test_adaptive_thinking_detection(self):
        """Opus 4.7 should use adaptive thinking, 4.6 should not."""
        assert AnthropicModelProvider._uses_adaptive_thinking("claude-opus-4-7") is True
        assert AnthropicModelProvider._uses_adaptive_thinking("claude-opus-4-6") is False
        assert AnthropicModelProvider._uses_adaptive_thinking("claude-sonnet-4-6") is False

    def test_temperature_support_detection(self):
        """Opus 4.7 should not support temperature, others should."""
        assert AnthropicModelProvider._supports_temperature("claude-opus-4-7") is False
        assert AnthropicModelProvider._supports_temperature("claude-opus-4-6") is True
        assert AnthropicModelProvider._supports_temperature("claude-sonnet-4-6") is True
        assert AnthropicModelProvider._supports_temperature("claude-haiku-4-5-20251001") is True

    def test_provider_type(self):
        """Test provider type identification."""
        provider = AnthropicModelProvider("test-key")
        assert provider.get_provider_type() == ProviderType.ANTHROPIC

    def test_supported_models_structure(self):
        """Test that MODEL_CAPABILITIES has the correct structure."""
        provider = AnthropicModelProvider("test-key")
        from providers.shared import ModelCapabilities

        # Check that all expected models are present
        assert "claude-opus-4-7" in provider.MODEL_CAPABILITIES
        assert "claude-opus-4-6" in provider.MODEL_CAPABILITIES
        assert "claude-sonnet-4-6" in provider.MODEL_CAPABILITIES
        assert "claude-haiku-4-5-20251001" in provider.MODEL_CAPABILITIES
        assert "claude-3-5-sonnet-20241022" in provider.MODEL_CAPABILITIES
        assert "claude-3-5-haiku-20241022" in provider.MODEL_CAPABILITIES

        # Check Opus 4.7 config
        opus47_config = provider.MODEL_CAPABILITIES["claude-opus-4-7"]
        assert isinstance(opus47_config, ModelCapabilities)
        assert opus47_config.context_window == 1_000_000
        assert opus47_config.max_output_tokens == 128_000
        assert opus47_config.supports_extended_thinking is True
        assert opus47_config.supports_temperature is False
        assert "opus" in opus47_config.aliases
        assert "claude-opus" in opus47_config.aliases

        # Check Opus 4.6 config
        opus46_config = provider.MODEL_CAPABILITIES["claude-opus-4-6"]
        assert isinstance(opus46_config, ModelCapabilities)
        assert opus46_config.context_window == 1_000_000
        assert opus46_config.supports_temperature is True


class TestAnthropicGenerateContent:
    """Test Anthropic generate_content with mocked SDK."""

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_basic(self, mock_anthropic_class):
        """Test basic text generation."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello, world!"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None

        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="Say hello",
            model_name="claude-haiku-4-5-20251001",
            temperature=0.5,
        )

        assert result.content == "Hello, world!"
        assert result.provider == ProviderType.ANTHROPIC
        assert result.model_name == "claude-haiku-4-5-20251001"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.metadata["stop_reason"] == "end_turn"

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_with_system_prompt(self, mock_anthropic_class):
        """Test that system prompt is sent as native parameter, not concatenated."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        provider.generate_content(
            prompt="Hello",
            model_name="haiku",
            system_prompt="You are a helpful assistant",
            temperature=0.5,
        )

        # Verify system prompt was passed as top-level parameter
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant"
        # And NOT concatenated into the user message
        user_content = call_kwargs["messages"][0]["content"]
        assert user_content[0]["text"] == "Hello"

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_with_budget_thinking(self, mock_anthropic_class):
        """Test budget_tokens thinking on Opus 4.6."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me reason about this..."

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "The answer is 42."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block, mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 100
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="What is the meaning of life?",
            model_name="claude-opus-4-6",
            thinking_mode="high",
        )

        # Verify budget_tokens thinking on 4.6
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"]["type"] == "enabled"
        # High = 67% of 128000 = 85760, clamped to min 1024
        assert call_kwargs["thinking"]["budget_tokens"] == 85760

        # Verify temperature forced to 1.0 for budget_tokens thinking
        assert call_kwargs["temperature"] == 1.0

        # Verify response
        assert result.content == "The answer is 42."
        assert result.metadata["thinking"] == "Let me reason about this..."

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_with_adaptive_thinking(self, mock_anthropic_class):
        """Test adaptive thinking on Opus 4.7."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Deep reasoning..."

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "The answer is 42."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block, mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 100
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="Complex problem",
            model_name="claude-opus-4-7",
            thinking_mode="high",
        )

        # Verify adaptive thinking on 4.7
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"]["type"] == "adaptive"
        assert call_kwargs["thinking"]["display"] == "summarized"
        assert call_kwargs["output_config"]["effort"] == "xhigh"

        # Verify NO temperature parameter sent (400 on Opus 4.7)
        assert "temperature" not in call_kwargs

        # Verify response
        assert result.content == "The answer is 42."
        assert result.metadata["thinking"] == "Deep reasoning..."
        assert result.metadata["thinking_mode"] == "high"

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_temperature_clamped(self, mock_anthropic_class):
        """Test that temperature > 1.0 is clamped to 1.0."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")

        # Anthropic models shouldn't normally get >1.0 but if they do, clamp it
        # The model's temperature constraint is "range" which defaults 0.0-2.0
        # but we clamp in generate_content()
        provider.generate_content(
            prompt="Test",
            model_name="haiku",
            temperature=0.8,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] <= 1.0

    @patch("providers.anthropic.anthropic.Anthropic")
    def test_generate_content_alias_resolution(self, mock_anthropic_class):
        """Test that aliases are resolved before API call."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="Test",
            model_name="opus",  # Alias -> claude-opus-4-7
        )

        # Verify the API received the canonical name
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-7"
        assert result.model_name == "claude-opus-4-7"


class TestAnthropicModelRestrictions:
    """Test Anthropic model restrictions."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"ANTHROPIC_ALLOWED_MODELS": "claude-opus-4-7"})
    def test_model_restrictions(self):
        """Test model restrictions functionality."""
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = AnthropicModelProvider("test-key")

        # Opus 4.7 should be allowed (including alias)
        assert provider.validate_model_name("claude-opus-4-7") is True
        assert provider.validate_model_name("opus") is True

        # Others should be blocked
        assert provider.validate_model_name("claude-opus-4-6") is False
        assert provider.validate_model_name("claude-sonnet-4-6") is False
        assert provider.validate_model_name("sonnet") is False

    @patch.dict(os.environ, {"ANTHROPIC_ALLOWED_MODELS": ""})
    def test_empty_restrictions_allows_all(self):
        """Test that empty restrictions allow all models."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = AnthropicModelProvider("test-key")

        assert provider.validate_model_name("claude-opus-4-6") is True
        assert provider.validate_model_name("claude-sonnet-4-6") is True
        assert provider.validate_model_name("claude-haiku-4-5-20251001") is True
        assert provider.validate_model_name("opus") is True
        assert provider.validate_model_name("sonnet") is True
        assert provider.validate_model_name("haiku") is True

    @patch.dict(os.environ, {"ANTHROPIC_ALLOWED_MODELS": "opus,claude-opus-4-6,sonnet,claude-sonnet-4-6"})
    def test_both_shorthand_and_full_name_allowed(self):
        """Test that aliases and canonical names can be allowed together."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = AnthropicModelProvider("test-key")

        assert provider.validate_model_name("opus") is True
        assert provider.validate_model_name("claude-opus-4-6") is True
        assert provider.validate_model_name("sonnet") is True
        assert provider.validate_model_name("claude-sonnet-4-6") is True


class TestClaudeCallingClaudeWarning:
    """Test Claude-calling-Claude detection."""

    @patch("providers.anthropic.get_cached_client_info")
    @patch("providers.anthropic.anthropic.Anthropic")
    def test_claude_client_warning(self, mock_anthropic_class, mock_client_info):
        """Test that metadata flag is set when MCP client is Claude."""
        mock_client_info.return_value = {
            "name": "claude-code",
            "version": "1.0.0",
            "friendly_name": "Claude",
        }

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="Test",
            model_name="haiku",
            temperature=0.5,
        )

        assert result.metadata.get("claude_calling_claude") is True

    @patch("providers.anthropic.get_cached_client_info")
    @patch("providers.anthropic.anthropic.Anthropic")
    def test_non_claude_client_no_warning(self, mock_anthropic_class, mock_client_info):
        """Test that no warning flag is set for non-Claude clients."""
        mock_client_info.return_value = {
            "name": "gemini-cli",
            "version": "2.0.0",
            "friendly_name": "Gemini",
        }

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicModelProvider("test-key")
        result = provider.generate_content(
            prompt="Test",
            model_name="haiku",
            temperature=0.5,
        )

        assert "claude_calling_claude" not in result.metadata
