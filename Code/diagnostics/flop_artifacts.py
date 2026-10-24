"""Exact FLOP table based on the number of logged estimator rounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_flop_artifacts(
    core: Any,
    run_dir: Path,
    tag: str,
    per_seed: list[dict[str, Any]],
    cfg: dict[str, Any],
    n_domains: int,
) -> None:
    n_params = int(per_seed[0].get("n_params") or 0)
    flops, base_flops = core.flop_table(cfg, n_params, tag, k=n_domains)
    methods_present = {method for result in per_seed for method in result["histories"]}
    if "DoGE" not in methods_present:
        flops = flops[~flops["method"].str.startswith("DoGE")]
    if "Skill-It" not in methods_present:
        flops = flops[~flops["method"].str.startswith("Skill-It")]
    if "DoGE" in methods_present:
        signal_rounds = max(
            len(result["histories"]["DoGE"].get("raw_scores", []))
            for result in per_seed
            if "DoGE" in result["histories"]
        )
        alignment_batches = (
            2 * n_domains * cfg.get("doge_est_batches", 2) * signal_rounds
        )
        tokens_per_batch = cfg["batch_size"] * cfg["context_length"]
        exact_doge_flops = (
            base_flops + 6 * n_params * alignment_batches * tokens_per_batch
        )
        doge_mask = flops["method"].str.startswith("DoGE")
        flops.loc[doge_mask, "flops"] = exact_doge_flops
        flops.loc[doge_mask, "note"] = (
            f"base + {alignment_batches} logged alignment micro-batches "
            f"across {signal_rounds} rounds"
        )
    flops.to_csv(run_dir / "flops.csv", index=False)
    try:
        markdown = flops.to_markdown(index=False)
    except Exception:
        markdown = flops.to_string(index=False)
    (run_dir / "flops.md").write_text(markdown, encoding="utf-8")
