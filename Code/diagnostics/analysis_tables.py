"""Convert experiment histories into analysis-ready diagnostic tables."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def l2_decomposition(weights: np.ndarray, is_synth: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    is_synth = np.asarray(is_synth, dtype=bool)
    uniform = 1.0 / len(weights)
    total_sq = float(np.square(weights - uniform).sum())
    between_sq = 0.0
    within_sq = 0.0
    for mask in (~is_synth, is_synth):
        group_mean = float(weights[mask].mean())
        between_sq += int(mask.sum()) * (group_mean - uniform) ** 2
        within_sq += float(np.square(weights[mask] - group_mean).sum())
    return {
        "l2": float(np.sqrt(total_sq)),
        "l2_sq": total_sq,
        "between_provenance_l2_sq": between_sq,
        "within_provenance_l2_sq": within_sq,
        "within_fraction_l2_sq": within_sq / total_sq if total_sq > 0 else 0.0,
    }


def weight_trajectory_table(
    per_seed: Iterable[dict[str, Any]], pretty: list[str], is_synth: np.ndarray
) -> pd.DataFrame:
    rows = []
    for result in per_seed:
        for method, history in result["histories"].items():
            for round_index, weights in zip(history["round"], history["p"]):
                for domain_index, weight in enumerate(np.asarray(weights, dtype=float)):
                    rows.append(
                        {
                            "method": method,
                            "seed": result["seed"],
                            "round": int(round_index),
                            "domain": pretty[domain_index],
                            "is_synth": bool(is_synth[domain_index]),
                            "weight": float(weight),
                        }
                    )
    return pd.DataFrame(rows)


def final_summary_table(
    per_seed: Iterable[dict[str, Any]], is_synth: np.ndarray
) -> pd.DataFrame:
    rows = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        for method, history in result["histories"].items():
            weights = np.asarray(history["p"][-1], dtype=float)
            rows.append(
                {
                    "method": method,
                    "seed": result["seed"],
                    "synthetic_share": float(weights[is_synth].sum()),
                    "real_share": float(weights[~is_synth].sum()),
                    **l2_decomposition(weights, is_synth),
                }
            )
    return pd.DataFrame(rows)


def doge_signal_tables(
    per_seed: Iterable[dict[str, Any]], pretty: list[str], is_synth: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domain_rows = []
    summary_rows = []
    block_rows = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        history = result["histories"].get("DoGE")
        if history is None or "raw_scores" not in history:
            continue
        seed = result["seed"]
        fields = zip(
            history["signal_round"],
            history["signal_step"],
            history["raw_scores"],
            history["normalized_scores"],
            history["score_normalizer"],
            history["alignment_to_real"],
            history["alignment_to_synthetic"],
            history["pre_update_p"],
            history["instantaneous_p"],
            history["p"],
        )
        for (
            round_index,
            step,
            raw,
            normalized,
            normalizer,
            align_real,
            align_synth,
            pre_p,
            inst_p,
            report_p,
        ) in fields:
            arrays = [
                np.asarray(value, dtype=float)
                for value in (
                    raw,
                    normalized,
                    align_real,
                    align_synth,
                    pre_p,
                    inst_p,
                    report_p,
                )
            ]
            raw, normalized, align_real, align_synth, pre_p, inst_p, report_p = arrays
            for domain_index, domain in enumerate(pretty):
                domain_rows.append(
                    {
                        "seed": seed,
                        "round": int(round_index),
                        "step": int(step),
                        "domain": domain,
                        "is_synth": bool(is_synth[domain_index]),
                        "raw_score": float(raw[domain_index]),
                        "normalized_score": float(normalized[domain_index]),
                        "score_normalizer": float(normalizer),
                        "alignment_to_real": float(align_real[domain_index]),
                        "alignment_to_synthetic": float(align_synth[domain_index]),
                        "normalized_alignment_to_real": float(
                            align_real[domain_index] / normalizer
                        ),
                        "normalized_alignment_to_synthetic": float(
                            align_synth[domain_index] / normalizer
                        ),
                        "pre_update_weight": float(pre_p[domain_index]),
                        "instantaneous_weight": float(inst_p[domain_index]),
                        "reported_time_average_weight": float(report_p[domain_index]),
                    }
                )
            real_mean = float(normalized[~is_synth].mean())
            synth_mean = float(normalized[is_synth].mean())
            summary_rows.append(
                {
                    "seed": seed,
                    "round": int(round_index),
                    "step": int(step),
                    "real_normalized_score_mean": real_mean,
                    "synthetic_normalized_score_mean": synth_mean,
                    "synthetic_minus_real_normalized_score": synth_mean - real_mean,
                    "real_instantaneous_share": float(inst_p[~is_synth].sum()),
                    "synthetic_instantaneous_share": float(inst_p[is_synth].sum()),
                    "synthetic_reported_share": float(report_p[is_synth].sum()),
                }
            )
            for train_label, train_mask in (
                ("real", ~is_synth),
                ("synthetic", is_synth),
            ):
                for target_label, values in (
                    ("real", align_real),
                    ("synthetic", align_synth),
                ):
                    block_rows.append(
                        {
                            "seed": seed,
                            "round": int(round_index),
                            "step": int(step),
                            "training_domain_provenance": train_label,
                            "target_gradient_provenance": target_label,
                            "mean_alignment": float(values[train_mask].mean()),
                            "mean_normalized_alignment": float(
                                values[train_mask].mean() / normalizer
                            ),
                        }
                    )
    return (
        pd.DataFrame(domain_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(block_rows),
    )
