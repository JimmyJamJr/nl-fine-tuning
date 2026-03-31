from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numeric_metrics(node: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_numeric_metrics(value, next_prefix))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        metrics[prefix] = float(node)
    return metrics


def find_first_json_with_key(directory: Path, required_key: str) -> Path | None:
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and required_key in payload:
            return path
    return None


def pick_primary_metric(
    metrics: dict[str, float], preferred_names: tuple[str, ...]
) -> tuple[str | None, float | None]:
    for name in preferred_names:
        if name in metrics:
            return name, metrics[name]
    if not metrics:
        return None, None
    first_key = sorted(metrics)[0]
    return first_key, metrics[first_key]
