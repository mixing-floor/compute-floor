"""Resumable seed persistence and final artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aioli_tables import aioli_block_table, aioli_signal_table
from .analysis_tables import (
    doge_signal_tables,
    final_summary_table,
    weight_trajectory_table,
)
from .flop_artifacts import write_flop_artifacts
from .seed_storage import save_npy_atomic
from .skillit_tables import skillit_graph_table, skillit_signal_tables
from .summaries import (
    aioli_provenance_summary,
    final_mixture_aggregate,
    score_seed_summary,
)


def _write_standard_aggregate(
    core: Any, run_dir: Path, per_seed: list[dict[str, Any]], is_synth: list[bool]
) -> None:
    aggregate = core.aggregate_over_seeds(per_seed, is_synth)
    rows = []
    for method, values in aggregate.items():
        for index, round_index in enumerate(values["rounds"]):
            rows.append(
                {
                    "method": method,
                    "round": int(round_index),
                    "synthetic_share_mean": values["share_mean"][index],
                    "synthetic_share_std": values["share_std"][index],
                    "l2_mean": values["l2_mean"][index],
                    "l2_std": values["l2_std"][index],
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "aggregate_trajectories.csv", index=False)
    if aggregate:
        core.save_error_band_figures(aggregate, str(run_dir / "figure"))


def _write_validation_losses(
    run_dir: Path,
    per_seed: list[dict[str, Any]],
    pretty: list[str],
    is_synth: list[bool],
) -> None:
    rows = []
    for result in per_seed:
        for method, history in result["histories"].items():
            if "val_losses" not in history:
                continue
            for round_index, losses in zip(history["round"], history["val_losses"]):
                for index, loss in enumerate(np.asarray(losses, dtype=float)):
                    if np.isnan(loss):
                        continue
                    rows.append(
                        {
                            "method": method,
                            "seed": result["seed"],
                            "round": int(round_index),
                            "domain": pretty[index],
                            "is_synth": bool(is_synth[index]),
                            "validation_loss": float(loss),
                        }
                    )
    if rows:
        pd.DataFrame(rows).to_csv(
            run_dir / "validation_loss_trajectories.csv", index=False
        )


def _write_test_losses(
    run_dir: Path,
    per_seed: list[dict[str, Any]],
    pretty: list[str],
    is_synth: list[bool],
) -> None:
    rows = []
    for result in per_seed:
        for method, losses in result.get("test_loss", {}).items():
            for index, loss in enumerate(np.asarray(losses, dtype=float)):
                rows.append(
                    {
                        "method": method,
                        "seed": result["seed"],
                        "domain": pretty[index],
                        "is_synth": bool(is_synth[index]),
                        "test_loss": float(loss),
                    }
                )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "test_losses_by_domain.csv", index=False)
    summary = frame.groupby(["method", "seed"], as_index=False)["test_loss"].mean()
    summary.to_csv(run_dir / "test_loss_seed_means.csv", index=False)
    provenance = (
        frame.assign(provenance=np.where(frame["is_synth"], "synthetic", "real"))
        .groupby(["method", "seed", "provenance"], as_index=False)["test_loss"]
        .mean()
    )
    provenance.to_csv(run_dir / "test_loss_provenance_means.csv", index=False)


def write_final_artifacts(
    core: Any,
    run_dir: Path,
    tag: str,
    per_seed: list[dict[str, Any]],
    pretty: list[str],
    is_synth: list[bool],
    cfg: dict[str, Any],
    seeds: list[int],
    scenario_name: str,
) -> list[str]:
    combined = {
        "per_seed": per_seed,
        "pretty": pretty,
        "is_synth": is_synth,
        "cfg": cfg,
        "seeds": seeds,
        "scenario": scenario_name,
        "tag": tag,
    }
    save_npy_atomic(run_dir / f"results_{tag}.npy", combined)
    is_synth_array = np.asarray(is_synth, dtype=bool)
    weight_trajectory_table(per_seed, pretty, is_synth_array).to_csv(
        run_dir / "weight_trajectories.csv", index=False
    )
    final_frame = final_summary_table(per_seed, is_synth_array)
    final_frame.to_csv(run_dir / "final_mixture_summary.csv", index=False)
    final_mixture_aggregate(final_frame).to_csv(
        run_dir / "final_mixture_aggregate.csv", index=False
    )
    doge_domains, doge_summary, doge_blocks = doge_signal_tables(
        per_seed, pretty, is_synth_array
    )
    if not doge_domains.empty:
        doge_domains.to_csv(run_dir / "doge_signals_by_domain.csv", index=False)
        doge_summary.to_csv(run_dir / "doge_provenance_summary.csv", index=False)
        doge_blocks.to_csv(run_dir / "doge_alignment_blocks.csv", index=False)
        score_seed_summary(doge_summary).to_csv(
            run_dir / "doge_seed_summary.csv", index=False
        )
    aioli = aioli_signal_table(per_seed, pretty, is_synth_array, cfg.get("ema"))
    if not aioli.empty:
        aioli.to_csv(run_dir / "aioli_column_sum_signals.csv", index=False)
        aioli_provenance_summary(aioli).to_csv(
            run_dir / "aioli_provenance_summary.csv", index=False
        )
        aioli_block_table(per_seed, is_synth_array, cfg.get("ema")).to_csv(
            run_dir / "aioli_interaction_blocks.csv", index=False
        )
    skillit_graph = skillit_graph_table(per_seed, pretty, is_synth_array)
    skillit_domains, skillit_summary = skillit_signal_tables(
        per_seed, pretty, is_synth_array
    )
    if not skillit_graph.empty:
        skillit_graph.to_csv(run_dir / "skillit_influence_graph.csv", index=False)
        skillit_domains.to_csv(run_dir / "skillit_signals_by_domain.csv", index=False)
        skillit_summary.to_csv(run_dir / "skillit_provenance_summary.csv", index=False)
        score_seed_summary(skillit_summary).to_csv(
            run_dir / "skillit_seed_summary.csv", index=False
        )
    _write_standard_aggregate(core, run_dir, per_seed, is_synth)
    _write_validation_losses(run_dir, per_seed, pretty, is_synth)
    _write_test_losses(run_dir, per_seed, pretty, is_synth)

    write_flop_artifacts(core, run_dir, tag, per_seed, cfg, len(pretty))
    return sorted(path.name for path in run_dir.iterdir() if path.is_file())
