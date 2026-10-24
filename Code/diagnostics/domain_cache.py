"""Materialize and reuse identical token blocks across diagnostic experiments."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _cache_payload(
    core: Any, cfg: dict[str, Any], scenario_name: str, split_seed: int
) -> dict[str, Any]:
    scenario = core.SCENARIOS[scenario_name]
    source_path = Path(core.__file__).resolve()
    return {
        "schema_version": 1,
        "scenario": scenario_name,
        "real_domains": scenario["real"],
        "synthetic_domains": scenario["synth"],
        "real_domain_sources": {
            key: core.REAL_DOMAINS[key] for key in scenario["real"]
        },
        "synthetic_domain_sources": {
            key: core.SYN_DOMAINS[key] for key in scenario["synth"]
        },
        "data_pipeline_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "tokenizer_name": cfg["tokenizer_name"],
        "context_length": cfg["context_length"],
        "train_tokens_per_domain": cfg["train_tokens_per_domain"],
        "val_tokens_per_domain": cfg["val_tokens_per_domain"],
        "test_tokens_per_domain": cfg["test_tokens_per_domain"],
        "split_seed": split_seed,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cached_domains(
    core: Any, cache_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    domains = []
    for entry in manifest["domains"]:
        for split, filename in entry["files"].items():
            actual = _file_sha256(cache_dir / filename)
            expected = entry["sha256"][split]
            if actual != expected:
                raise RuntimeError(
                    f"Corrupt domain cache file {filename}: {actual} != {expected}"
                )
        shards = {
            split: core.DomainShard(
                np.load(cache_dir / entry["files"][split], allow_pickle=False)
            )
            for split in ("train", "val", "test")
        }
        domains.append(
            {
                **shards,
                "name": entry["name"],
                "pretty": entry["pretty"],
                "is_synth": bool(entry["is_synth"]),
            }
        )
    return domains


def _save_domain_cache(
    cache_dir: Path, domains: list[dict[str, Any]], payload: dict[str, Any]
) -> None:
    build_dir = cache_dir.with_name(
        f"{cache_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    build_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    try:
        for index, domain in enumerate(domains):
            files = {}
            checksums = {}
            shapes = {}
            for split in ("train", "val", "test"):
                filename = f"domain_{index:02d}_{split}.npy"
                arrays = np.asarray(domain[split].arrays, dtype=np.int32)
                np.save(build_dir / filename, arrays, allow_pickle=False)
                files[split] = filename
                checksums[split] = _file_sha256(build_dir / filename)
                shapes[split] = list(arrays.shape)
            entries.append(
                {
                    "name": domain["name"],
                    "pretty": domain["pretty"],
                    "is_synth": bool(domain["is_synth"]),
                    "files": files,
                    "sha256": checksums,
                    "shapes": shapes,
                }
            )
        manifest = {**payload, "fingerprint": _fingerprint(payload), "domains": entries}
        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        build_dir.replace(cache_dir)
    except Exception:
        # Leave the uniquely named build directory for diagnosis; it is never treated as valid cache.
        raise


def load_or_build_domains(
    core: Any,
    tokenizer: Any,
    load_domain_dataset: Any,
    cfg: dict[str, Any],
    scenario_name: str,
    split_seed: int,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], Path, bool]:
    """Return domains, cache directory, and whether a new cache was built."""
    payload = _cache_payload(core, cfg, scenario_name, split_seed)
    digest = _fingerprint(payload)
    cache_dir = cache_root / f"{scenario_name.lower()}_{digest[:12]}"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != digest:
            raise RuntimeError(f"Domain-cache fingerprint mismatch at {cache_dir}")
        return _load_cached_domains(core, cache_dir, manifest), cache_dir, False

    if cache_dir.exists():
        raise RuntimeError(
            f"Incomplete domain cache at {cache_dir}. Move it aside and rerun; no files were deleted."
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    domains = core.build_scenario_domains(
        tokenizer,
        load_domain_dataset,
        core.SCENARIOS[scenario_name],
        cfg,
        split_seed=split_seed,
    )
    _save_domain_cache(cache_dir, domains, payload)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    return _load_cached_domains(core, cache_dir, manifest), cache_dir, True
