#!/usr/bin/env python3
"""Regenerate the 3-panel central figure (f_central.png) from the aggregated CSVs.

Fixes the reviewer-reported bug in panel (c): the y-limit clipped the upper error
bars for Group-DRO (4.14+0.41), DoReMi-style (3.96+0.56) and Skill-It (4.19+0.37).
The panel now sets its top limit from max(mean + std) with explicit headroom.

Usage:
    python Code/make_central_figure.py --out-dir Code/aioli_runs --fig Paper/latex/figures/f_central.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

# Palette matching the published f_central.png
COLOR = {
    "Group-DRO":    "#d62728",
    "DoReMi-style": "#ff7f0e",
    "Stratified":   "#7f7f7f",
    "Aioli":        "#2ca02c",
    "DoGE":         "#1f77b4",
    "Skill-It":     "#9467bd",
}
ORDER = ["Group-DRO", "DoReMi-style", "Stratified", "Aioli", "DoGE", "Skill-It"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="Code/aioli_runs")
    ap.add_argument("--tag", default="HEAVY_S2_9R_9S")
    ap.add_argument("--fig", default="Paper/latex/figures/f_central.png")
    a = ap.parse_args()

    out = Path(a.out_dir)
    ss = pd.read_csv(out / f"agg_{a.tag}_synth_share.csv")
    tl = pd.read_csv(out / f"agg_{a.tag}_test_loss.csv")
    tl = tl[tl["eval"] == "test_loss"].set_index("method")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    present = [m for m in ORDER if m in set(ss["method"])]

    # ---- (a) synthetic share ----
    ax = axes[0]
    for m in present:
        s = ss[ss["method"] == m].sort_values("round")
        ax.plot(s["round"], s["synth_share_mean"], color=COLOR[m], marker="o", markersize=3.5, label=m)
        if s["synth_share_std"].gt(0).any():
            ax.fill_between(s["round"], s["synth_share_mean"] - s["synth_share_std"],
                            s["synth_share_mean"] + s["synth_share_std"], color=COLOR[m], alpha=0.15)
    ax.axhline(0.5, color="grey", lw=0.9, ls=":")
    ax.set_title("(a) Synthetic share")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Synthetic share")

    # ---- (b) L2 from uniform ----
    ax = axes[1]
    for m in present:
        s = ss[ss["method"] == m].sort_values("round")
        ax.plot(s["round"], s["l2_mean"], color=COLOR[m], marker="o", markersize=3.5, label=m)
        if s["l2_std"].gt(0).any():
            ax.fill_between(s["round"], s["l2_mean"] - s["l2_std"], s["l2_mean"] + s["l2_std"],
                            color=COLOR[m], alpha=0.15)
    ax.set_title(r"(b) $L_2$ from uniform")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel(r"$L_2$ distance from uniform")

    # ---- (c) held-out test loss (error bars must NOT be clipped) ----
    ax = axes[2]
    methods = [m for m in ORDER if m in tl.index]
    means = [float(tl.loc[m, "test_loss_mean"]) for m in methods]
    errs = [float(tl.loc[m, "test_loss_std"]) for m in methods]
    ax.bar(range(len(methods)), means, yerr=errs, capsize=4,
           color=[COLOR[m] for m in methods], alpha=0.9,
           error_kw={"linewidth": 1.2, "ecolor": "black"})
    if "Stratified" in tl.index:
        ax.axhline(float(tl.loc["Stratified", "test_loss_mean"]), color="grey", lw=0.9, ls=":")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_title("(c) Held-out test loss")
    ax.set_ylabel("Held-out test loss")

    # The fix: bound the axis by the largest mean+std, with headroom so no bar is cut.
    top = max(m + e for m, e in zip(means, errs))
    bot = min(m - e for m, e in zip(means, errs))
    pad = 0.10 * (top - bot)
    ax.set_ylim(max(0.0, bot - pad), top + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    figp = Path(a.fig); figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figp}")
    print(f"  panel (c) y-limits: {ax.get_ylim()}  (max mean+std = {top:.3f})")


if __name__ == "__main__":
    main()
