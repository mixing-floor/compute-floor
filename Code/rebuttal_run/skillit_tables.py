"""Analysis-ready exports for the logged Skill-It graph and scores."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def skillit_graph_table(
    per_seed: Iterable[dict[str, Any]], pretty: list[str], is_synth: np.ndarray
) -> pd.DataFrame:
    rows = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        history = result["histories"].get("Skill-It")
        if history is None or "skill_graph" not in history:
            continue
        graph = np.asarray(history["skill_graph"], dtype=float)
        for train_index, train_domain in enumerate(pretty):
            for eval_index, eval_domain in enumerate(pretty):
                rows.append(
                    {
                        "seed": result["seed"],
                        "training_domain": train_domain,
                        "training_is_synth": bool(is_synth[train_index]),
                        "evaluated_domain": eval_domain,
                        "evaluated_is_synth": bool(is_synth[eval_index]),
                        "influence": float(graph[train_index, eval_index]),
                    }
                )
    return pd.DataFrame(rows)


def skillit_signal_tables(
    per_seed: Iterable[dict[str, Any]], pretty: list[str], is_synth: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        history = result["histories"].get("Skill-It")
        if history is None or "raw_scores" not in history:
            continue
        fields = zip(
            history["signal_round"],
            history["signal_step"],
            history["val_losses"],
            history["raw_scores"],
            history["normalized_scores"],
            history["score_normalizer"],
            history["pre_update_p"],
            history["instantaneous_p"],
        )
        for (
            round_index,
            step,
            losses,
            raw,
            normalized,
            normalizer,
            pre_p,
            inst_p,
        ) in fields:
            arrays = [
                np.asarray(value, dtype=float)
                for value in (losses, raw, normalized, pre_p, inst_p)
            ]
            losses, raw, normalized, pre_p, inst_p = arrays
            for index, domain in enumerate(pretty):
                rows.append(
                    {
                        "seed": result["seed"],
                        "round": int(round_index),
                        "step": int(step),
                        "domain": domain,
                        "is_synth": bool(is_synth[index]),
                        "validation_loss": float(losses[index]),
                        "raw_score": float(raw[index]),
                        "normalized_score": float(normalized[index]),
                        "score_normalizer": float(normalizer),
                        "pre_update_weight": float(pre_p[index]),
                        "instantaneous_weight": float(inst_p[index]),
                    }
                )
            real_mean = float(normalized[~is_synth].mean())
            synth_mean = float(normalized[is_synth].mean())
            summaries.append(
                {
                    "seed": result["seed"],
                    "round": int(round_index),
                    "step": int(step),
                    "real_normalized_score_mean": real_mean,
                    "synthetic_normalized_score_mean": synth_mean,
                    "synthetic_minus_real_normalized_score": synth_mean - real_mean,
                    "synthetic_instantaneous_share": float(inst_p[is_synth].sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries)
