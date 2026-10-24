"""Atomic JSON/NumPy persistence and resume compatibility checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npy_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def save_seed_result(
    path: Path, result: dict[str, Any], metadata: dict[str, Any]
) -> None:
    result["_rebuttal_metadata"] = metadata
    save_npy_atomic(path, result)


def load_seed_result(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    result = np.load(path, allow_pickle=True).item()
    metadata = result.get("_rebuttal_metadata", {})
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Cannot resume incompatible seed artifact {path}: {mismatches}"
        )
    return result
