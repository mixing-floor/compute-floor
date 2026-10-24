"""Command-line contract and immutable run-plan construction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .experiment_specs import changed_heavy_fields


def parse_args(experiments: dict[str, Any], description: str) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("experiment", choices=sorted(experiments))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument(
        "--output-root", type=Path, default=repo_root / "runs" / "diagnostics"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=repo_root / ".cache" / "diagnostics"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse compatible completed seed files."
    )
    parser.add_argument(
        "--skip-eval-test", action="store_true", help="Skip final held-out test loss."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit training without CUDA (debug only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without loading data/models.",
    )
    return parser.parse_args()


def build_plan(
    core: Any,
    args: argparse.Namespace,
    spec: Any,
    scenario: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    run_dir = args.output_root.resolve() / spec.output_slug
    return {
        "experiment": args.experiment,
        "name": spec.name,
        "purpose": spec.purpose,
        "scenario": scenario,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "run_all_methods": spec.run_all_methods,
        "rng_protocol": spec.rng_protocol,
        "evaluate_test": not args.skip_eval_test,
        "output_directory": run_dir,
        "cache_root": args.cache_root.resolve(),
        "config": cfg,
        "changes_from_submitted_heavy": changed_heavy_fields(core, cfg),
    }


def resolve_device(core: Any, allow_cpu: bool) -> str:
    device = "cuda" if core.torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is required. Use --allow-cpu only for a tiny debugging run."
        )
    core.torch.backends.cuda.matmul.allow_tf32 = True
    return device
