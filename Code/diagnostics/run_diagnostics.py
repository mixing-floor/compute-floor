"""Run either signal-level diagnostic experiment on a GPU server."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diagnostics.artifact_io import write_final_artifacts  # type: ignore
    from diagnostics.cli_options import build_plan, parse_args, resolve_device  # type: ignore
    from diagnostics.domain_cache import load_or_build_domains  # type: ignore
    from diagnostics.experiment_execution import run_doge_only_seed, run_full_seed  # type: ignore
    from diagnostics.experiment_specs import (  # type: ignore
        EXPERIMENTS,
        build_config,
        validate_spec_config,
    )
    from diagnostics.runtime_metadata import (
        environment,
        git_commit,
        package_sha256,
        seed_metadata,
        sha256,
        utc_now,
    )  # type: ignore
    from diagnostics.source_runner import SOURCE_RUNNER_PATH, core  # type: ignore
    from diagnostics.seed_storage import (  # type: ignore
        load_seed_result,
        save_seed_result,
        write_json_atomic,
    )
else:
    from .artifact_io import write_final_artifacts
    from .cli_options import build_plan, parse_args, resolve_device
    from .domain_cache import load_or_build_domains
    from .experiment_execution import run_doge_only_seed, run_full_seed
    from .experiment_specs import (
        EXPERIMENTS,
        build_config,
        validate_spec_config,
    )
    from .source_runner import SOURCE_RUNNER_PATH, core
    from .runtime_metadata import (
        environment,
        git_commit,
        package_sha256,
        seed_metadata,
        sha256,
        utc_now,
    )
    from .seed_storage import load_seed_result, save_seed_result, write_json_atomic


SCENARIO_NAME = "S2_9R_9S"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(EXPERIMENTS, __doc__ or "Signal-level diagnostic experiments")
    spec = EXPERIMENTS[args.experiment]
    cfg = build_config(core, spec)
    validate_spec_config(core, spec, cfg)
    plan = build_plan(core, args, spec, SCENARIO_NAME, cfg)
    run_dir = plan["output_directory"]
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return

    device = resolve_device(core, args.allow_cpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_seed_files = list(run_dir.glob("seed_*.npy"))
    if existing_seed_files and not args.resume:
        raise FileExistsError(
            f"Seed artifacts already exist in {run_dir}. Use --resume or choose another --output-root."
        )

    manifest = {
        **plan,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "source_runner": str(SOURCE_RUNNER_PATH),
        "source_runner_sha256": sha256(SOURCE_RUNNER_PATH),
        "diagnostics_code_sha256": package_sha256(Path(__file__).resolve().parent),
        "git_commit": git_commit(SOURCE_RUNNER_PATH.parents[1]),
        "environment": environment(core, device),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        domains, cache_dir, cache_built = load_or_build_domains(
            core,
            tokenizer,
            core.make_dataset_factory(tokenizer),
            cfg,
            SCENARIO_NAME,
            args.split_seed,
            args.cache_root.resolve(),
        )
        is_synth = [bool(domain["is_synth"]) for domain in domains]
        pretty = [domain["pretty"] for domain in domains]
        cache_manifest = json.loads(
            (cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest.update(
            {
                "domain_cache": str(cache_dir),
                "domain_cache_built": cache_built,
                "domain_cache_fingerprint": cache_manifest["fingerprint"],
            }
        )
        write_json_atomic(run_dir / "run_manifest.json", manifest)

        per_seed = []
        for seed in args.seeds:
            seed_path = run_dir / f"seed_{seed}.npy"
            expected = seed_metadata(
                SOURCE_RUNNER_PATH,
                Path(__file__).resolve().parent,
                args.experiment,
                SCENARIO_NAME,
                args.split_seed,
                seed,
                cfg,
                not args.skip_eval_test,
            )
            if seed_path.exists() and args.resume:
                logging.info("Loading completed seed %s from %s", seed, seed_path)
                result = load_seed_result(seed_path, expected)
            else:
                logging.info("Starting experiment=%s seed=%s", args.experiment, seed)
                if spec.run_all_methods:
                    result = run_full_seed(
                        core,
                        domains,
                        tokenizer,
                        cfg,
                        device,
                        seed,
                        not args.skip_eval_test,
                    )
                else:
                    result = run_doge_only_seed(
                        core,
                        domains,
                        tokenizer,
                        cfg,
                        device,
                        seed,
                        not args.skip_eval_test,
                    )
                result["seed"] = seed
                save_seed_result(seed_path, result, expected)
                logging.info("Saved completed seed %s to %s", seed, seed_path)
            per_seed.append(result)

        files = write_final_artifacts(
            core,
            run_dir,
            spec.output_slug,
            per_seed,
            pretty,
            is_synth,
            cfg,
            args.seeds,
            SCENARIO_NAME,
        )
        manifest.update(
            {
                "status": "COMPLETED",
                "completed_at_utc": utc_now(),
                "output_files": files,
            }
        )
        write_json_atomic(run_dir / "run_manifest.json", manifest)
        logging.info("Completed %s. Results: %s", args.experiment, run_dir)
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "failed_at_utc": utc_now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(run_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
