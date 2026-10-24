"""Runtime provenance recorded alongside diagnostic experiment artifacts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_sha256(package_dir: Path) -> str:
    """Hash production Python sources so resume cannot mix logger versions."""
    digest = hashlib.sha256()
    files = sorted(
        path for path in package_dir.glob("*.py") if not path.name.startswith("test_")
    )
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(core: Any, device: str) -> dict[str, Any]:
    package_versions = {}
    for package in ("transformers", "datasets", "accelerate", "numpy", "pandas"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": core.torch.__version__,
        "cuda_runtime": core.torch.version.cuda,
        "device": device,
        "packages": package_versions,
    }
    if core.torch.cuda.is_available():
        info["gpu"] = core.torch.cuda.get_device_name(0)
        info["gpu_count"] = core.torch.cuda.device_count()
    return info


def seed_metadata(
    source_runner_path: Path,
    package_dir: Path,
    experiment: str,
    scenario: str,
    split_seed: int,
    seed: int,
    cfg: dict[str, Any],
    evaluate_test: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "seed": seed,
        "scenario": scenario,
        "split_seed": split_seed,
        "model_name": cfg["model_name"],
        "tokenizer_name": cfg["tokenizer_name"],
        "evaluate_test": evaluate_test,
        "source_runner_sha256": sha256(source_runner_path),
        "diagnostics_code_sha256": package_sha256(package_dir),
    }
