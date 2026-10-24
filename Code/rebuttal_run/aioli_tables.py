"""Export Aioli's applied interaction matrices and EMA-consumed scores."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _applied_matrices(history: dict[str, Any], ema: float | None):
    ema_matrix = None
    applied_count = max(0, len(history["p"]) - 1)
    for index, matrix in enumerate(history["A"][:applied_count]):
        matrix = np.asarray(matrix, dtype=float)
        ema_matrix = (
            matrix
            if ema_matrix is None or ema is None
            else (1 - ema) * matrix + ema * ema_matrix
        )
        yield index, matrix, ema_matrix


def aioli_signal_table(
    per_seed: Iterable[dict[str, Any]],
    pretty: list[str],
    is_synth: np.ndarray,
    ema: float | None,
) -> pd.DataFrame:
    rows = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        history = result["histories"].get("Aioli")
        if history is None:
            continue
        for index, matrix, ema_matrix in _applied_matrices(history, ema):
            raw_scores = matrix.sum(axis=0)
            consumed_scores = ema_matrix.sum(axis=0)
            for domain_index, domain in enumerate(pretty):
                rows.append(
                    {
                        "seed": result["seed"],
                        "round": int(history["round"][index]),
                        "domain": domain,
                        "is_synth": bool(is_synth[domain_index]),
                        "raw_column_sum": float(raw_scores[domain_index]),
                        "ema_column_sum_consumed": float(consumed_scores[domain_index]),
                    }
                )
    return pd.DataFrame(rows)


def aioli_block_table(
    per_seed: Iterable[dict[str, Any]], is_synth: np.ndarray, ema: float | None
) -> pd.DataFrame:
    rows = []
    is_synth = np.asarray(is_synth, dtype=bool)
    for result in per_seed:
        history = result["histories"].get("Aioli")
        if history is None:
            continue
        for index, matrix, ema_matrix in _applied_matrices(history, ema):
            for eval_label, eval_mask in (("real", ~is_synth), ("synthetic", is_synth)):
                for train_label, train_mask in (
                    ("real", ~is_synth),
                    ("synthetic", is_synth),
                ):
                    block = np.ix_(eval_mask, train_mask)
                    rows.append(
                        {
                            "seed": result["seed"],
                            "round": int(history["round"][index]),
                            "evaluated_provenance": eval_label,
                            "training_domain_provenance": train_label,
                            "raw_mean_interaction": float(matrix[block].mean()),
                            "ema_mean_interaction_consumed": float(
                                ema_matrix[block].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)
