"""Sync PAL per-provider model configs from LiteLLM + overlay.

Source of truth layers:
  1. ``conf/litellm_models.json`` — committed snapshot of
     https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
  2. ``conf/pal_overrides.json`` — hand-edited overlay containing PAL-specific
     fields (intelligence_score, aliases, max_thinking_tokens, ...) plus any
     per-model overrides of LiteLLM's mappable fields.

Merge rule: overlay wins unconditionally. Fields sourced from LiteLLM only
appear in the generated output if the overlay does not supply them.

Subcommands:
  sync       — regenerate conf/*_models.json from snapshot + overlay (default)
  refresh    — redownload snapshot from upstream, then run sync
  check      — run sync in-memory, exit 1 if output differs from committed files
  bootstrap  — one-shot: build overlay from current conf/*_models.json, applying
               known data fixes. Writes conf/pal_overrides.json.

Run via ``devenv shell -- python scripts/sync_litellm.py`` so the resolved
Python and dependencies match the project.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
SNAPSHOT_PATH = CONF_DIR / "litellm_models.json"
OVERLAY_PATH = CONF_DIR / "pal_overrides.json"
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/" "model_prices_and_context_window.json"


LITELLM_TO_PAL_FIELDS: dict[str, str] = {
    "max_input_tokens": "context_window",
    "max_output_tokens": "max_output_tokens",
    "supports_reasoning": "supports_extended_thinking",
    "supports_function_calling": "supports_function_calling",
    "supports_vision": "supports_images",
    "supports_response_schema": "supports_json_mode",
    "supports_system_messages": "supports_system_prompts",
}


# Fields that are allowed in an overlay entry (on top of the PAL dataclass field names).
OVERLAY_META_FIELDS: set[str] = {"litellm_key"}


# Providers in emission order (determines key order in pal_overrides.json when bootstrapped).
PROVIDER_FILES: dict[str, str] = {
    "anthropic": "anthropic_models.json",
    "openai": "openai_models.json",
    "gemini": "gemini_models.json",
    "xai": "xai_models.json",
    "azure": "azure_models.json",
    "dial": "dial_models.json",
    "openrouter": "openrouter_models.json",
    "custom": "custom_models.json",
}


# Known data drift to fix when bootstrapping: (provider, model_name) -> set of
# fields where LiteLLM's value should replace PAL's current value.
KNOWN_DATA_FIXES: dict[tuple[str, str], set[str]] = {
    ("anthropic", "claude-sonnet-4-6"): {"context_window"},
    # Entire GPT-5 family: PAL had 400K, OpenAI/LiteLLM agree on 272K input tokens.
    ("openai", "gpt-5"): {"context_window"},
    ("openai", "gpt-5-mini"): {"context_window"},
    ("openai", "gpt-5-nano"): {"context_window"},
    ("openai", "gpt-5-codex"): {"context_window"},
    ("openai", "gpt-5.1"): {"context_window"},
    ("openai", "gpt-5.2-pro"): {"context_window"},
    # NOTE: intentionally do NOT fix o3.supports_extended_thinking — PAL's
    # False is deliberate: o-series uses OpenAI reasoning_effort, not the
    # thinking-budget mechanism PAL's flag controls. LiteLLM's supports_reasoning
    # is broader than PAL's supports_extended_thinking.
    ("openai", "o3"): {"max_output_tokens"},
    ("gemini", "gemini-2.5-flash"): {"max_output_tokens"},
    ("gemini", "gemini-2.5-pro"): {"max_output_tokens"},
}


@dataclass(frozen=True)
class LoadedSnapshot:
    data: dict[str, dict[str, Any]]
    size_bytes: int

    def lookup(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        return self.data.get(key)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> LoadedSnapshot:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Snapshot at {path} must be a JSON object")
    data.pop("sample_spec", None)
    return LoadedSnapshot(data=data, size_bytes=len(raw))


def load_overlay(path: Path = OVERLAY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Overlay missing at {path}. Run 'python scripts/sync_litellm.py bootstrap' first.")
    with path.open() as f:
        data = json.load(f)
    if "providers" not in data or not isinstance(data["providers"], dict):
        raise ValueError(f"Overlay at {path} must have a 'providers' object")
    return data


_INT_PAL_FIELDS: set[str] = {"context_window", "max_output_tokens"}


def map_litellm_entry(litellm_entry: dict[str, Any]) -> dict[str, Any]:
    """Extract PAL-shaped fields from a LiteLLM entry.

    LiteLLM occasionally stores token counts as floats (e.g. grok-4-1-fast
    entries have ``max_input_tokens: 2000000.0``). Coerce to int so the
    generated PAL config stays integer-typed.
    """

    mapped: dict[str, Any] = {}
    for litellm_field, pal_field in LITELLM_TO_PAL_FIELDS.items():
        if litellm_field not in litellm_entry:
            continue
        value = litellm_entry[litellm_field]
        if pal_field in _INT_PAL_FIELDS and isinstance(value, float) and value.is_integer():
            value = int(value)
        mapped[pal_field] = value
    return mapped


# Stable emission order mirroring the existing conf/*_models.json convention.
# Fields not listed here retain insertion order at the end.
_EMIT_ORDER: tuple[str, ...] = (
    "model_name",
    "friendly_name",
    "aliases",
    "intelligence_score",
    "description",
    "deployment",
    "deployment_name",
    "context_window",
    "max_output_tokens",
    "max_thinking_tokens",
    "supports_extended_thinking",
    "supports_system_prompts",
    "supports_streaming",
    "supports_function_calling",
    "supports_json_mode",
    "supports_images",
    "supports_temperature",
    "use_openai_response_api",
    "default_reasoning_effort",
    "allow_code_generation",
    "max_image_size_mb",
    "temperature_constraint",
    "provider",
)


def _ordered(entry: dict[str, Any]) -> dict[str, Any]:
    ordered = {k: entry[k] for k in _EMIT_ORDER if k in entry}
    for k, v in entry.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def merge_model(
    model_name: str,
    overlay_entry: dict[str, Any],
    snapshot: LoadedSnapshot,
) -> dict[str, Any]:
    """Produce a PAL model dict by merging LiteLLM base + overlay."""

    litellm_entry = snapshot.lookup(overlay_entry.get("litellm_key"))
    merged: dict[str, Any] = {"model_name": model_name}

    if litellm_entry is not None:
        merged.update(map_litellm_entry(litellm_entry))

    for key, value in overlay_entry.items():
        if key in OVERLAY_META_FIELDS:
            continue
        merged[key] = value

    return _ordered(merged)


def build_provider_config(
    provider_name: str,
    provider_block: dict[str, Any],
    snapshot: LoadedSnapshot,
    *,
    readme: dict[str, Any] | None = None,
) -> dict[str, Any]:
    models_block = provider_block.get("models", {})
    if not isinstance(models_block, dict):
        raise ValueError(f"provider '{provider_name}' must have a dict of models, got {type(models_block).__name__}")

    emitted_models: list[dict[str, Any]] = []
    for model_name, overlay_entry in models_block.items():
        if not isinstance(overlay_entry, dict):
            raise ValueError(f"overlay for {provider_name}/{model_name} must be a dict")
        emitted_models.append(merge_model(model_name, overlay_entry, snapshot))

    config: dict[str, Any] = {}
    if readme is not None:
        config["_README"] = readme
    config["_generated"] = {
        "by": "scripts/sync_litellm.py",
        "from": [
            "conf/litellm_models.json (LiteLLM snapshot)",
            "conf/pal_overrides.json (PAL overlay)",
        ],
        "note": "Do not edit by hand. Edit the overlay and rerun the sync.",
    }
    config["models"] = emitted_models
    return config


def read_existing_readme(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    readme = data.get("_README")
    return readme if isinstance(readme, dict) else None


def render_configs(
    overlay: dict[str, Any],
    snapshot: LoadedSnapshot,
) -> dict[str, dict[str, Any]]:
    rendered: dict[str, dict[str, Any]] = {}
    for provider_name, filename in PROVIDER_FILES.items():
        provider_block = overlay["providers"].get(provider_name)
        if provider_block is None:
            continue
        # Skip providers with no models in the overlay — the existing file may
        # be a user-customized template (e.g. azure_models.json with
        # _example_models) and we shouldn't clobber it with a blank regeneration.
        if not provider_block.get("models"):
            continue
        readme = read_existing_readme(CONF_DIR / filename)
        rendered[filename] = build_provider_config(provider_name, provider_block, snapshot, readme=readme)
    return rendered


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_configs(rendered: dict[str, dict[str, Any]]) -> list[Path]:
    written: list[Path] = []
    for filename, config in rendered.items():
        path = CONF_DIR / filename
        path.write_text(dump_json(config))
        written.append(path)
    return written


def check_configs(rendered: dict[str, dict[str, Any]]) -> list[Path]:
    """Return list of files whose on-disk content differs from rendered output."""

    drift: list[Path] = []
    for filename, config in rendered.items():
        path = CONF_DIR / filename
        expected = dump_json(config)
        actual = path.read_text() if path.exists() else ""
        if expected != actual:
            drift.append(path)
    return drift


def download_snapshot(dest: Path = SNAPSHOT_PATH) -> int:
    with urllib.request.urlopen(LITELLM_URL) as response:
        body = response.read()
    # Validate JSON before overwriting.
    data = json.loads(body)
    if not isinstance(data, dict) or "sample_spec" not in data:
        raise ValueError("Downloaded snapshot did not contain a sample_spec — refusing to overwrite.")
    dest.write_bytes(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    return len(body)


# ---------------------------------------------------------------------------
# Bootstrap (one-shot migration)
# ---------------------------------------------------------------------------


# Heuristic candidates for finding the right LiteLLM key for a PAL model_name.
def _litellm_key_candidates(provider: str, model_name: str) -> list[str]:
    candidates = [model_name]
    provider_prefixes = {
        "anthropic": ["anthropic/"],
        "openai": ["openai/"],
        "gemini": ["gemini/", "vertex_ai/"],
        "xai": ["xai/", "x-ai/"],
    }
    for prefix in provider_prefixes.get(provider, []):
        candidates.append(prefix + model_name)
    return candidates


def _find_litellm_key(provider: str, model_name: str, snapshot: LoadedSnapshot) -> str | None:
    # For openrouter and custom providers, do not try to match LiteLLM keys;
    # their configs stay overlay-only for v1.
    if provider in {"openrouter", "custom", "azure", "dial"}:
        return None
    for candidate in _litellm_key_candidates(provider, model_name):
        if candidate in snapshot.data:
            return candidate
    return None


# Fields always kept in the overlay (PAL-specific or hand-crafted).
_ALWAYS_OVERLAY_FIELDS: set[str] = {
    "intelligence_score",
    "aliases",
    "max_thinking_tokens",
    "allow_code_generation",
    "friendly_name",
    "description",
    "supports_temperature",
    "supports_streaming",
    "temperature_constraint",
    "max_image_size_mb",
    "default_reasoning_effort",
    "use_openai_response_api",
    "provider",  # openrouter entries use this to route between openrouter vs custom
}


def _bootstrap_model_entry(
    provider: str,
    model_name: str,
    pal_entry: dict[str, Any],
    snapshot: LoadedSnapshot,
) -> dict[str, Any]:
    litellm_key = _find_litellm_key(provider, model_name, snapshot)
    litellm_entry = snapshot.lookup(litellm_key) if litellm_key else None
    litellm_mapped = map_litellm_entry(litellm_entry) if litellm_entry else {}

    fixes_for_model = KNOWN_DATA_FIXES.get((provider, model_name), set())

    overlay_entry: dict[str, Any] = {"litellm_key": litellm_key}

    for field, pal_value in pal_entry.items():
        if field == "model_name":
            continue
        if field in _ALWAYS_OVERLAY_FIELDS:
            overlay_entry[field] = pal_value
            continue
        if field in litellm_mapped:
            litellm_value = litellm_mapped[field]
            if pal_value == litellm_value:
                # PAL matches LiteLLM — trust LiteLLM going forward; no override.
                continue
            if field in fixes_for_model:
                # Known data issue — take LiteLLM's value, drop PAL's override.
                continue
            # PAL genuinely disagrees — preserve as explicit override.
            overlay_entry[field] = pal_value
            continue
        # Field is not in LiteLLM's mapping at all (e.g., deployment). Keep.
        overlay_entry[field] = pal_value

    # When no LiteLLM source, everything PAL had must live in the overlay.
    if litellm_key is None:
        for field, pal_value in pal_entry.items():
            if field == "model_name":
                continue
            overlay_entry.setdefault(field, pal_value)

    return overlay_entry


def bootstrap_overlay(snapshot: LoadedSnapshot) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "_README": {
            "description": "PAL overlay on top of LiteLLM baseline.",
            "rule": "Overlay wins unconditionally. LiteLLM fills in any field the overlay doesn't set.",
            "edit": "Hand-edit this file. Then run: devenv shell -- python scripts/sync_litellm.py",
            "sources": {
                "litellm": "conf/litellm_models.json (committed snapshot of LiteLLM's model_prices_and_context_window.json)",
            },
            "fields": {
                "litellm_key": "LiteLLM entry key to use as the base layer. Set to null to opt out (overlay must supply all capability fields).",
                "...everything else": "Passes through to the generated PAL config, overriding any LiteLLM value.",
            },
        },
        "providers": {},
    }

    for provider_name, filename in PROVIDER_FILES.items():
        path = CONF_DIR / filename
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        models = data.get("models") or []
        if not isinstance(models, list):
            continue

        provider_block: dict[str, Any] = {"file": f"conf/{filename}", "models": {}}
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("model_name")
            if not name:
                continue
            provider_block["models"][name] = _bootstrap_model_entry(provider_name, name, model, snapshot)
        # Skip providers with no models (e.g. azure_models.json template).
        if not provider_block["models"]:
            continue
        overlay["providers"][provider_name] = provider_block

    return overlay


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _cmd_sync(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    overlay = load_overlay()
    rendered = render_configs(overlay, snapshot)
    written = write_configs(rendered)
    print(f"Wrote {len(written)} config files from {snapshot.size_bytes:,} bytes of LiteLLM snapshot + overlay.")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    overlay = load_overlay()
    rendered = render_configs(overlay, snapshot)
    drift = check_configs(rendered)
    if drift:
        print("Drift detected in the following files:", file=sys.stderr)
        for path in drift:
            print(f"  {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "\nRun 'devenv shell -- python scripts/sync_litellm.py' to regenerate.",
            file=sys.stderr,
        )
        return 1
    print(f"No drift — {len(rendered)} generated files match the overlay.")
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    size = download_snapshot()
    print(f"Downloaded snapshot ({size:,} bytes) to {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
    return _cmd_sync(args)


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    overlay = bootstrap_overlay(snapshot)
    if OVERLAY_PATH.exists() and not args.force:
        print(
            f"Overlay already exists at {OVERLAY_PATH.relative_to(REPO_ROOT)}. " f"Pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1
    OVERLAY_PATH.write_text(dump_json(overlay))
    model_count = sum(len(p["models"]) for p in overlay["providers"].values())
    with_litellm = sum(
        1 for p in overlay["providers"].values() for entry in p["models"].values() if entry.get("litellm_key")
    )
    print(
        f"Wrote overlay for {model_count} models "
        f"({with_litellm} with LiteLLM source, {model_count - with_litellm} overlay-only) "
        f"to {OVERLAY_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Sync PAL model configs from LiteLLM + overlay.").splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd")

    sync_p = sub.add_parser("sync", help="Regenerate conf/*_models.json (default).")
    sync_p.set_defaults(func=_cmd_sync)

    check_p = sub.add_parser("check", help="Verify no drift; exit 1 if files need regeneration.")
    check_p.set_defaults(func=_cmd_check)

    refresh_p = sub.add_parser("refresh", help="Redownload LiteLLM snapshot, then sync.")
    refresh_p.set_defaults(func=_cmd_refresh)

    bootstrap_p = sub.add_parser(
        "bootstrap",
        help="One-shot: build conf/pal_overrides.json from current conf/*_models.json.",
    )
    bootstrap_p.add_argument("--force", action="store_true", help="Overwrite existing overlay.")
    bootstrap_p.set_defaults(func=_cmd_bootstrap)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args.func = _cmd_sync
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
