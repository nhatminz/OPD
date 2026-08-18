from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_name = config.pop("_base_", None)
    if base_name is not None:
        config = deep_merge(load_config(path.parent / base_name), config)
    config["_config_path"] = str(path)
    return config


def load_with_overlays(
    path: str | Path, overlays: list[str | Path] | None = None
) -> dict[str, Any]:
    config = load_config(path)
    for overlay_path in overlays or []:
        overlay = load_config(overlay_path)
        overlay.pop("_config_path", None)
        config = deep_merge(config, overlay)
    config["_config_path"] = str(Path(path).resolve())
    config["_overlays"] = [str(Path(item).resolve()) for item in overlays or []]
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected dotted.key=value")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(
                    f"Cannot set {dotted_key!r}: {part!r} is not a mapping"
                )
        target[parts[-1]] = value
    return result


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
