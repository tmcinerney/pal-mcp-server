"""Google Vertex AI provider built on the Gemini provider implementation."""

from __future__ import annotations

import logging
from typing import ClassVar

from google import genai
from google.genai import types

from .gemini import GeminiModelProvider
from .registries.vertex_ai import VertexAIModelRegistry
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class VertexAIModelProvider(GeminiModelProvider):
    """Thin Vertex AI wrapper that reuses the Gemini request pipeline.

    Same SDK (google-genai), different auth: ADC instead of API key.
    Overrides only client init, provider type, registry, and friendly name.
    """

    REGISTRY_CLASS = VertexAIModelRegistry
    FRIENDLY_NAME = "Vertex AI"
    # AIDEV-NOTE: Own class-level state prevents inheriting GeminiModelProvider's
    # cached registry and capability map via MRO.
    _registry = None
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    def __init__(
        self,
        api_key: str,
        *,
        project: str | None = None,
        location: str | None = None,
        **kwargs,
    ) -> None:
        self._project = project
        self._location = location or "us-central1"
        super().__init__(api_key, **kwargs)

    @property
    def client(self):
        """Lazy initialization of Vertex AI client using ADC."""
        if self._client is None:
            http_options_kwargs: dict[str, object] = {}
            if self._timeout_override is not None:
                http_options_kwargs["timeout"] = self._timeout_override

            client_kwargs: dict[str, object] = {
                "vertexai": True,
                "project": self._project,
                "location": self._location,
            }

            if http_options_kwargs:
                client_kwargs["http_options"] = types.HttpOptions(**http_options_kwargs)

            logger.debug(
                "Initializing Vertex AI client: project=%s location=%s",
                self._project,
                self._location,
            )
            self._client = genai.Client(**client_kwargs)
        return self._client

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.VERTEX_AI


# Load registry data at import time for registry consumers
VertexAIModelProvider._ensure_registry()
