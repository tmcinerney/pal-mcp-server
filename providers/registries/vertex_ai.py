"""Registry loader for Vertex AI model capabilities."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class VertexAIModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/vertex_ai_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="VERTEX_AI_MODELS_CONFIG_PATH",
            default_filename="vertex_ai_models.json",
            provider=ProviderType.VERTEX_AI,
            friendly_prefix="Vertex AI ({model})",
            config_path=config_path,
        )
