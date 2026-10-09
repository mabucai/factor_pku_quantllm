"""Combine public experiment parameters with private local data locations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    config = json.loads((BASE_DIR / "research_settings.json").read_text(encoding="utf-8"))
    local_path = BASE_DIR / "config.json"
    if local_path.exists():
        try:
            local = json.loads(local_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError("Local config.json must contain valid JSON.") from None
        if not isinstance(local, dict):
            raise ValueError("Local config.json must be a JSON object.")
        # Existing local configs can still override experiment parameters.
        config.update(local)

    lab_path = os.environ.get("FACTOR_LAB_PATH") or config.get("lab_path")
    if not isinstance(lab_path, str) or not lab_path.strip():
        raise ValueError(
            "Set lab_path in the ignored local config.json (copy config.example.json), "
            "or set FACTOR_LAB_PATH. Never commit local configuration."
        )
    cache_dir = os.environ.get("FACTOR_CACHE_DIR") or config.get("cache_dir") or "runs/cache"
    if not isinstance(cache_dir, str):
        raise ValueError("cache_dir must be a string.")
    config["lab_path"] = str((BASE_DIR / lab_path).resolve())
    config["cache_dir"] = str((BASE_DIR / cache_dir).resolve())
    return config
