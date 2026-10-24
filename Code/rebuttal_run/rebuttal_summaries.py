"""Compact summaries designed for direct use in the rebuttal audit."""

from __future__ import annotations

import pandas as pd


def score_seed_summary(round_summary: pd.DataFrame) -> pd.DataFrame:
    if round_summary.empty:
        return pd.DataFrame()
    gap = "synthetic_minus_real_normalized_score"
    rows = []
    for seed, frame in round_summary.groupby("seed", sort=True):
        values = frame[gap].to_numpy(dtype=float)
        row = {
            "seed": seed,
            "rounds": len(values),
            "mean_synthetic_minus_real_normalized_score": float(values.mean()),
            "std_synthetic_minus_real_normalized_score": float(values.std()),
            "realward_rounds": int((values < 0).sum()),
            "syntheticward_rounds": int((values > 0).sum()),
            "zero_rounds": int((values == 0).sum()),
        }
        for column in (
            "synthetic_instantaneous_share",
            "synthetic_reported_share",
        ):
            if column in frame:
                row[f"final_{column}"] = float(frame.iloc[-1][column])
        rows.append(row)
    return pd.DataFrame(rows)


def aioli_provenance_summary(aioli: pd.DataFrame) -> pd.DataFrame:
    if aioli.empty:
        return pd.DataFrame()
    rows = []
    for (seed, round_index), frame in aioli.groupby(["seed", "round"], sort=True):
        real = frame[~frame["is_synth"]]
        synthetic = frame[frame["is_synth"]]
        raw_real = float(real["raw_column_sum"].mean())
        raw_synthetic = float(synthetic["raw_column_sum"].mean())
        ema_real = float(real["ema_column_sum_consumed"].mean())
        ema_synthetic = float(synthetic["ema_column_sum_consumed"].mean())
        rows.append(
            {
                "seed": seed,
                "round": round_index,
                "raw_real_mean": raw_real,
                "raw_synthetic_mean": raw_synthetic,
                "raw_synthetic_minus_real": raw_synthetic - raw_real,
                "ema_real_mean": ema_real,
                "ema_synthetic_mean": ema_synthetic,
                "ema_synthetic_minus_real_consumed": ema_synthetic - ema_real,
            }
        )
    return pd.DataFrame(rows)


def final_mixture_aggregate(final_frame: pd.DataFrame) -> pd.DataFrame:
    if final_frame.empty:
        return pd.DataFrame()
    metrics = [
        "synthetic_share",
        "l2",
        "between_provenance_l2_sq",
        "within_provenance_l2_sq",
        "within_fraction_l2_sq",
    ]
    grouped = final_frame.groupby("method", sort=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=0).fillna(0.0).add_suffix("_std")
    return mean.join(std).reset_index()
