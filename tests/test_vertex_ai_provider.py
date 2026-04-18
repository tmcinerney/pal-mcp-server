"""Tests for Vertex AI provider implementation."""

import os
from unittest.mock import MagicMock, patch

from providers.shared import ModelResponse, ProviderType
from providers.vertex_ai import VertexAIModelProvider


class TestVertexAIProvider:
    """Test Vertex AI provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def test_initialization(self):
        """Test provider initialization with project and location."""
        provider = VertexAIModelProvider(
            api_key="",
            project="my-project",
            location="us-central1",
        )
        assert provider._project == "my-project"
        assert provider._location == "us-central1"
        assert provider._client is None  # Lazy init

    def test_initialization_default_location(self):
        """Test provider defaults to us-central1 when location is not specified."""
        provider = VertexAIModelProvider(api_key="", project="my-project")
        assert provider._location == "us-central1"

    def test_provider_type(self):
        """Test provider type is VERTEX_AI."""
        provider = VertexAIModelProvider(api_key="", project="my-project")
        assert provider.get_provider_type() == ProviderType.VERTEX_AI

    def test_friendly_name(self):
        """Test friendly name is Vertex AI."""
        provider = VertexAIModelProvider(api_key="", project="my-project")
        assert provider.FRIENDLY_NAME == "Vertex AI"

    def test_model_validation(self):
        """Test model name validation against Vertex AI registry."""
        provider = VertexAIModelProvider(api_key="", project="my-project")

        # Canonical model names
        assert provider.validate_model_name("gemini-2.5-flash") is True
        assert provider.validate_model_name("gemini-2.5-pro") is True
        assert provider.validate_model_name("gemini-3.1-pro-preview") is True
        assert provider.validate_model_name("gemini-3-flash-preview") is True

        # Invalid models
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("claude-sonnet") is False

    def test_alias_resolution(self):
        """Test vertex-prefixed alias resolution."""
        provider = VertexAIModelProvider(api_key="", project="my-project")

        assert provider._resolve_model_name("vertex-pro") == "gemini-3.1-pro-preview"
        assert provider._resolve_model_name("vertex-flash") == "gemini-2.5-flash"
        assert provider._resolve_model_name("vertex-flashlite") == "gemini-3.1-flash-lite-preview"
        assert provider._resolve_model_name("vertex-flash3") == "gemini-3-flash-preview"

        # Canonical names pass through
        assert provider._resolve_model_name("gemini-2.5-flash") == "gemini-2.5-flash"

    def test_capabilities(self):
        """Test model capabilities are loaded from registry."""
        provider = VertexAIModelProvider(api_key="", project="my-project")

        caps = provider.get_capabilities("gemini-2.5-flash")
        assert caps.supports_images is True
        assert caps.supports_temperature is True
        assert caps.supports_extended_thinking is True

        # Non-thinking model
        caps_lite = provider.get_capabilities("gemini-2.5-flash-lite")
        assert caps_lite.supports_extended_thinking is False

    @patch("providers.vertex_ai.genai.Client")
    def test_lazy_client_init(self, mock_client_class):
        """Test client is lazily initialized with vertexai=True."""
        provider = VertexAIModelProvider(
            api_key="",
            project="test-project",
            location="global",
        )
        assert provider._client is None

        # Access client triggers initialization
        _ = provider.client
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["location"] == "global"

    @patch("providers.vertex_ai.genai.Client")
    def test_generate_content_returns_vertex_identity(self, mock_client_class):
        """Test generate_content returns ModelResponse with Vertex AI identity."""
        mock_response = MagicMock()
        mock_response.text = "Hello from Vertex"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.candidates[0].safety_ratings = None
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = VertexAIModelProvider(
            api_key="",
            project="test-project",
            location="us-central1",
        )

        result = provider.generate_content(
            prompt="test prompt",
            model_name="gemini-2.5-flash",
        )

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello from Vertex"
        assert result.provider == ProviderType.VERTEX_AI
        assert result.friendly_name == "Vertex AI"

    @patch("providers.vertex_ai.genai.Client")
    def test_generate_content_system_instruction(self, mock_client_class):
        """Test system_prompt is passed as system_instruction, not concatenated."""
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.candidates[0].safety_ratings = None
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = VertexAIModelProvider(
            api_key="",
            project="test-project",
            location="us-central1",
        )

        provider.generate_content(
            prompt="user query",
            model_name="gemini-2.5-flash",
            system_prompt="you are a helpful assistant",
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs[1]["config"]
        assert config.system_instruction == "you are a helpful assistant"

        # User prompt should NOT contain system prompt
        contents = call_kwargs[1]["contents"]
        user_text = contents[0]["parts"][0]["text"]
        assert user_text == "user query"

    @patch.dict(os.environ, {"VERTEX_AI_ALLOWED_MODELS": "gemini-2.5-flash,gemini-2.5-pro"})
    def test_restrictions(self):
        """Test model restrictions are applied correctly."""
        provider = VertexAIModelProvider(api_key="", project="my-project")

        # Allowed models
        assert provider.validate_model_name("gemini-2.5-flash") is True
        assert provider.validate_model_name("gemini-2.5-pro") is True

        # Restricted models
        assert provider.validate_model_name("gemini-2.0-flash") is False
