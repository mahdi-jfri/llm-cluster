from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_API_KEYS_PATH = "api-keys.json"


def load_api_keys(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load provider API keys from a local JSON file."""

    resolved_path = _resolve_api_keys_path(path)
    if not resolved_path.exists():
        return {}

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved_path}.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{resolved_path} must contain a JSON object.")

    return payload


def get_api_key(
    provider: str,
    *,
    path: str | os.PathLike[str] | None = None,
) -> str | None:
    """Return a provider API key from `api-keys.json`."""

    payload = load_api_keys(path)
    value = payload.get(provider)
    if isinstance(value, str) and value:
        return value
    return None


def _resolve_api_keys_path(path: str | os.PathLike[str] | None) -> Path:
    configured_path = path or os.getenv("API_KEYS_FILE") or DEFAULT_API_KEYS_PATH
    return Path(configured_path).expanduser()
