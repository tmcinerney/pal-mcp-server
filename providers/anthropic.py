"""Anthropic Claude model provider implementation."""

import base64
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

import anthropic

from utils.client_info import get_cached_client_info
from utils.image_utils import validate_image

from .base import ModelProvider
from .registries.anthropic import AnthropicModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class AnthropicModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Anthropic integration built on the official Anthropic SDK.

    The provider uses the Anthropic Messages API natively, supporting system
    prompts as a top-level parameter, extended thinking (budget tokens for
    4.6 models, adaptive thinking for 4.7+), and image content blocks.
    Includes a Claude-calling-Claude warning when the MCP client is itself Claude.
    """

    REGISTRY_CLASS = AnthropicModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # AIDEV-NOTE: Opus 4.7+ uses adaptive thinking with effort levels instead of budget_tokens.
    # budget_tokens returns 400 on 4.7+. The THINKING_BUDGETS map is only used for <=4.6 models.

    # Thinking budget percentages of model's max_thinking_tokens (for <=4.6 models)
    THINKING_BUDGETS = {
        "minimal": 0.005,  # 0.5% of max - minimal thinking for fast responses
        "low": 0.08,  # 8% of max - light reasoning tasks
        "medium": 0.33,  # 33% of max - balanced reasoning (default)
        "high": 0.67,  # 67% of max - complex analysis
        "max": 1.0,  # 100% of max - full thinking budget
    }

    # Adaptive thinking effort levels for Opus 4.7+ models
    # Maps PAL thinking modes to Anthropic effort parameter values
    ADAPTIVE_EFFORT_LEVELS = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",  # xhigh recommended for coding/agentic work
        "max": "max",
    }

    @staticmethod
    def _uses_adaptive_thinking(model_name: str) -> bool:
        """Return True if the model uses adaptive thinking instead of budget_tokens.

        Opus 4.7+ requires adaptive thinking; budget_tokens returns a 400 error.
        Opus 4.6 and Sonnet 4.6 support both but adaptive is recommended.
        """
        return "opus-4-7" in model_name or "opus-4-8" in model_name or "opus-4-9" in model_name

    @staticmethod
    def _supports_temperature(model_name: str) -> bool:
        """Return True if the model accepts temperature/sampling parameters.

        Opus 4.7+ returns 400 on any non-default temperature, top_p, or top_k.
        """
        # Models that do NOT support temperature
        if "opus-4-7" in model_name or "opus-4-8" in model_name or "opus-4-9" in model_name:
            return False
        return True

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic provider with API key and optional base URL."""
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._client = None
        self._base_url = kwargs.get("base_url", None)
        self._invalidate_capability_cache()

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import httpx

            client_kwargs = {
                "api_key": self.api_key,
                # AIDEV-NOTE: PAL is a synchronous MCP server that blocks on responses.
                # Large thinking budgets can take minutes; default 10min SDK timeout is too low.
                "timeout": httpx.Timeout(timeout=600.0, connect=30.0),
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
                logger.debug("Initializing Anthropic client with custom endpoint: %s", self._base_url)
            self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ANTHROPIC

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
        thinking_mode: str = "medium",
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using an Anthropic Claude model.

        Args:
            prompt: The main user prompt/query to send to the model
            model_name: Canonical model name or alias (e.g., "claude-opus-4-6", "opus")
            system_prompt: Optional system instructions (sent as native system parameter)
            temperature: Controls randomness (0.0-1.0), default 1.0
            max_output_tokens: Optional maximum tokens to generate
            thinking_mode: Thinking budget level ("minimal", "low", "medium", "high", "max")
            images: Optional list of image paths or data URLs
            **kwargs: Additional keyword arguments

        Returns:
            ModelResponse with generated content, usage stats, and metadata
        """
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        resolved_model_name = self._resolve_model_name(model_name)

        # Build content blocks
        content = [{"type": "text", "text": prompt}]

        # Add images if provided and model supports vision
        if images and capabilities.supports_images:
            for image_path in images:
                try:
                    image_part = self._process_image(image_path)
                    if image_part:
                        content.append(image_part)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    continue
        elif images and not capabilities.supports_images:
            logger.warning(f"Model {resolved_model_name} does not support images, ignoring {len(images)} image(s)")

        messages = [{"role": "user", "content": content}]

        # Prepare API call kwargs
        api_kwargs: dict = {
            "model": resolved_model_name,
            "messages": messages,
        }

        # System prompt as native top-level parameter
        if system_prompt:
            api_kwargs["system"] = system_prompt

        # Handle extended thinking
        use_thinking = capabilities.supports_extended_thinking and thinking_mode
        effective_thinking_mode = thinking_mode if use_thinking else None
        is_adaptive = self._uses_adaptive_thinking(resolved_model_name)

        if use_thinking:
            effective_max_tokens = max_output_tokens or capabilities.max_output_tokens or 16000
            if is_adaptive:
                # AIDEV-NOTE: Opus 4.7+ uses adaptive thinking with effort levels.
                # budget_tokens returns 400 on these models.
                effort = self.ADAPTIVE_EFFORT_LEVELS.get(thinking_mode, "high")
                api_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
                api_kwargs["output_config"] = {"effort": effort}
            else:
                # Opus 4.6, Sonnet 4.6: use budget_tokens (still functional, deprecated)
                budget_pct = self.THINKING_BUDGETS.get(thinking_mode, self.THINKING_BUDGETS["medium"])
                budget_tokens = max(1024, int(capabilities.max_thinking_tokens * budget_pct))
                # AIDEV-NOTE: Anthropic requires max_tokens > budget_tokens
                if budget_tokens >= effective_max_tokens:
                    effective_max_tokens = budget_tokens + 4096
                api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            api_kwargs["max_tokens"] = effective_max_tokens
        else:
            api_kwargs["max_tokens"] = max_output_tokens or capabilities.max_output_tokens or 4096

        # Temperature handling - model-specific
        if not is_adaptive and self._supports_temperature(resolved_model_name):
            effective_temp = capabilities.get_effective_temperature(temperature)
            if use_thinking:
                # AIDEV-NOTE: Anthropic requires temperature=1 when budget_tokens thinking is enabled
                effective_temp = 1.0
            if effective_temp is not None:
                effective_temp = min(effective_temp, 1.0)
                api_kwargs["temperature"] = effective_temp
        # AIDEV-NOTE: Opus 4.7+ returns 400 on any temperature param - omit entirely

        # Claude-calling-Claude detection
        claude_calling_claude = False
        client_info = get_cached_client_info()
        if client_info and client_info.get("friendly_name") == "Claude":
            claude_calling_claude = True
            logger.warning(
                "Claude-calling-Claude detected: MCP client '%s' is routing to Anthropic provider. "
                "Consider whether this request could be handled directly by the host model.",
                client_info.get("name", "unknown"),
            )

        # Retry logic
        max_retries = 4
        retry_delays = [1.0, 3.0, 5.0, 8.0]
        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self.client.messages.create(**api_kwargs)

            # Extract text content from response blocks
            text_parts = []
            thinking_content = None
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_content = block.thinking

            content_text = "\n".join(text_parts) if text_parts else ""
            usage = self._extract_usage(response)

            metadata = {
                "thinking_mode": effective_thinking_mode,
                "stop_reason": response.stop_reason,
            }
            if thinking_content is not None:
                metadata["thinking"] = thinking_content
            if claude_calling_claude:
                metadata["claude_calling_claude"] = True

            return ModelResponse(
                content=content_text,
                usage=usage,
                model_name=resolved_model_name,
                friendly_name="Claude",
                provider=ProviderType.ANTHROPIC,
                metadata=metadata,
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix=f"Anthropic API ({resolved_model_name})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = (
                f"Anthropic API error for model {resolved_model_name} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            )
            raise RuntimeError(error_msg) from exc

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Anthropic response."""
        usage = {}
        try:
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "input_tokens", None)
                output_tokens = getattr(response.usage, "output_tokens", None)

                if input_tokens is not None:
                    usage["input_tokens"] = input_tokens
                if output_tokens is not None:
                    usage["output_tokens"] = output_tokens
                if input_tokens is not None and output_tokens is not None:
                    usage["total_tokens"] = input_tokens + output_tokens

                # Cache token metrics (useful for future prompt caching)
                cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
                cache_read = getattr(response.usage, "cache_read_input_tokens", None)
                if cache_creation is not None:
                    usage["cache_creation_tokens"] = cache_creation
                if cache_read is not None:
                    usage["cache_read_tokens"] = cache_read
        except (AttributeError, TypeError):
            pass
        return usage

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an Anthropic API error should be retried.

        Anthropic-specific: 429 rate limits ARE retryable (unlike the base class
        which excludes 429s by default). Authentication errors are never retried.
        """
        # Use Anthropic SDK exception types when available
        if isinstance(error, anthropic.AuthenticationError):
            return False

        if isinstance(error, anthropic.RateLimitError):
            return True

        if isinstance(error, anthropic.APIConnectionError):
            return True

        if isinstance(error, anthropic.APIStatusError):
            status = getattr(error, "status_code", None)
            if status in (408, 429, 500, 502, 503, 504):
                return True
            return False

        # Fall back to string matching for unexpected error types
        error_str = str(error).lower()
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "overloaded",
            "500",
            "502",
            "503",
            "504",
        ]
        return any(indicator in error_str for indicator in retryable_indicators)

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def _process_image(self, image_path: str) -> dict | None:
        """Process an image for the Anthropic Messages API."""
        try:
            image_bytes, mime_type = validate_image(image_path)

            if image_path.startswith("data:"):
                _, data = image_path.split(",", 1)
            else:
                data = base64.b64encode(image_bytes).decode()

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": data,
                },
            }
        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Model preference
    # ------------------------------------------------------------------

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get Anthropic's preferred model for a given category.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        capability_map = self.get_all_model_capabilities()

        def find_best(candidates: list[str]) -> str | None:
            """Return best model from candidates (sorted for consistency)."""
            return sorted(candidates, reverse=True)[0] if candidates else None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer models with thinking support, Opus first
            opus_thinking = [
                m
                for m in allowed_models
                if "opus" in m and m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if opus_thinking:
                return find_best(opus_thinking)

            any_thinking = [
                m for m in allowed_models if m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if any_thinking:
                return find_best(any_thinking)

            # Fall back to Opus without thinking
            opus_models = [m for m in allowed_models if "opus" in m]
            if opus_models:
                return find_best(opus_models)

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Haiku for speed
            haiku_models = [m for m in allowed_models if "haiku" in m]
            if haiku_models:
                return find_best(haiku_models)

            # Fall back to Sonnet
            sonnet_models = [m for m in allowed_models if "sonnet" in m]
            if sonnet_models:
                return find_best(sonnet_models)

        # BALANCED or fallback: prefer Sonnet, then Opus, then anything
        sonnet_models = [m for m in allowed_models if "sonnet" in m]
        if sonnet_models:
            return find_best(sonnet_models)

        opus_models = [m for m in allowed_models if "opus" in m]
        if opus_models:
            return find_best(opus_models)

        return find_best(allowed_models)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Anthropic client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


# Load registry data at import time for registry consumers
AnthropicModelProvider._ensure_registry()
