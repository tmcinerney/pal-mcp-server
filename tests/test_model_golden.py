"""Golden capability assertions + drift detection for the LiteLLM-backed registries.

These tests lock down a small set of high-signal model capabilities so that an
accidental edit to ``conf/pal_overrides.json`` or an unreviewed LiteLLM snapshot
refresh cannot silently change what PAL exposes to clients.

They complement ``scripts/sync_litellm.py`` by:

* Asserting critical capability numbers on representative models.
* Re-running the sync pipeline in memory and asserting the generated output
  matches the committed ``conf/*_models.json`` files.

See the ticket-level rationale in
``https://linear.app/tmcinerney/issue/TMC-100``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sync_litellm  # type: ignore[import-not-found]  # noqa: E402 — sys.path modified above

from providers.registries.anthropic import AnthropicModelRegistry  # noqa: E402
from providers.registries.gemini import GeminiModelRegistry  # noqa: E402
from providers.registries.openai import OpenAIModelRegistry  # noqa: E402
from providers.registries.xai import XAIModelRegistry  # noqa: E402

# (registry class, model name, expected capability attrs)
GOLDEN_CAPABILITIES = [
    (
        AnthropicModelRegistry,
        "claude-opus-4-7",
        {
            "context_window": 1_000_000,
            "max_output_tokens": 128_000,
            "supports_extended_thinking": True,
            "supports_temperature": False,
            "intelligence_score": 20,
            "allow_code_generation": True,
        },
    ),
    (
        AnthropicModelRegistry,
        "claude-sonnet-4-6",
        {
            # Previously 200_000 in PAL's hand-curated config — LiteLLM has 1M,
            # which matches Anthropic's 1M-context beta for Sonnet.
            "context_window": 1_000_000,
            "max_output_tokens": 64_000,
            "supports_extended_thinking": True,
            "intelligence_score": 16,
        },
    ),
    (
        OpenAIModelRegistry,
        "gpt-5",
        {
            # Previously 400_000 in PAL — LiteLLM (and OpenAI) have gpt-5 at 272K.
            "context_window": 272_000,
            "max_output_tokens": 128_000,
            "intelligence_score": 16,
        },
    ),
    (
        OpenAIModelRegistry,
        "o3",
        {
            "context_window": 200_000,
            # Previously 65_536 in PAL — LiteLLM has the correct 100_000 max output.
            "max_output_tokens": 100_000,
            # Intentionally False: PAL's supports_extended_thinking means
            # "takes a thinking budget"; o-series uses reasoning_effort instead.
            "supports_extended_thinking": False,
            "supports_temperature": False,
        },
    ),
    (
        GeminiModelRegistry,
        "gemini-2.5-flash",
        {
            "context_window": 1_048_576,
            # Previously 65_536 in PAL — LiteLLM has the correct 65_535.
            "max_output_tokens": 65_535,
            "supports_extended_thinking": True,
        },
    ),
    (
        GeminiModelRegistry,
        "gemini-2.5-pro",
        {
            "context_window": 1_048_576,
            "max_output_tokens": 65_535,
        },
    ),
    (
        XAIModelRegistry,
        "grok-4",
        {
            "context_window": 256_000,
        },
    ),
]


@pytest.mark.parametrize(
    "registry_cls,model_name,expected",
    [(cls, name, exp) for cls, name, exp in GOLDEN_CAPABILITIES],
    ids=[f"{cls.__name__}:{name}" for cls, name, _ in GOLDEN_CAPABILITIES],
)
def test_golden_capabilities(registry_cls, model_name, expected):
    registry = registry_cls()
    caps = registry.resolve(model_name)
    assert caps is not None, f"{model_name} not resolvable via {registry_cls.__name__}"
    mismatches = {
        field: (getattr(caps, field), expected_value)
        for field, expected_value in expected.items()
        if getattr(caps, field) != expected_value
    }
    assert not mismatches, f"{model_name} capability drift: {mismatches}"


# Core aliases that downstream tools depend on. If any of these change, callers
# will silently fail to resolve their preferred short name.
ALIAS_GOLDENS = [
    (AnthropicModelRegistry, "opus", "claude-opus-4-7"),
    (AnthropicModelRegistry, "sonnet", "claude-sonnet-4-6"),
    (OpenAIModelRegistry, "gpt5", "gpt-5"),
    (GeminiModelRegistry, "flash", "gemini-2.5-flash"),
    (XAIModelRegistry, "grok", "grok-4"),
]


@pytest.mark.parametrize(
    "registry_cls,alias,expected_name",
    ALIAS_GOLDENS,
    ids=[f"{cls.__name__}:{alias}" for cls, alias, _ in ALIAS_GOLDENS],
)
def test_alias_resolution(registry_cls, alias, expected_name):
    registry = registry_cls()
    caps = registry.resolve(alias)
    assert caps is not None, f"alias {alias!r} not resolvable"
    assert caps.model_name == expected_name


def test_no_sync_drift():
    """Regenerating the configs in-memory must match the committed files.

    Fails when:
      * Someone edits ``conf/*_models.json`` by hand without updating the overlay.
      * Someone edits ``conf/pal_overrides.json`` without rerunning the sync.
      * The ``conf/litellm_models.json`` snapshot was updated without a resync.
    """

    snapshot = sync_litellm.load_snapshot()
    overlay = sync_litellm.load_overlay()
    rendered = sync_litellm.render_configs(overlay, snapshot)

    drift: list[str] = []
    for filename, config in rendered.items():
        path = sync_litellm.CONF_DIR / filename
        expected = sync_litellm.dump_json(config)
        actual = path.read_text() if path.exists() else ""
        if expected != actual:
            drift.append(filename)

    assert not drift, (
        "Generated configs drift from committed files: "
        f"{drift}. Run `devenv shell -- python scripts/sync_litellm.py` to regenerate."
    )


def test_overlay_structure():
    """Every overlay entry must have a litellm_key field (possibly null)."""

    overlay = sync_litellm.load_overlay()
    missing: list[str] = []
    for provider_name, provider_block in overlay["providers"].items():
        for model_name, entry in provider_block["models"].items():
            if "litellm_key" not in entry:
                missing.append(f"{provider_name}/{model_name}")
    assert not missing, (
        f"overlay entries missing litellm_key: {missing}. " "Set to null for custom/overlay-only models."
    )
