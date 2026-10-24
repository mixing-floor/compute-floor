#!/usr/bin/env python3
"""
Standalone figure generator for the Aioli data-mixing study.

Reads the CSV artifacts written by run_experiment.py and regenerates the
paper's appendix figures, with multi-seed error bands where applicable.

Six figure types are produced per (config, scenario) tag:
  1. comparative_synth_share  — total synthetic share vs. round, all methods, ±1σ
  2. divergence               — L2 from uniform vs. round, all methods, ±1σ
  3. final_weights            — horizontal grouped bar chart of final domain weights
  4. <method>_weight_evolution — per-domain weight trajectories per method, ±1σ
  5. <method>_val_loss        — per-domain val-loss trajectories (Aioli + any other
                                 method that logged val_losses), ±1σ
  6. test_loss                — held-out test-loss bar chart (only if --eval-test was run)

Required CSVs (always written by run_experiment.py):
  agg_<tag>_synth_share.csv
  agg_<tag>_final_weights.csv

Required CSVs for figures 4 & 5:
  agg_<tag>_weight_trajectories.csv
  agg_<tag>_val_loss_trajectories.csv

Optional CSV (written only when --eval-test is passed to run_experiment.py):
  agg_<tag>_test_loss.csv

Usage:
    # all configs and scenarios found in Results/main (writes to Results/figures)
    python Code/visualize_results.py

    # single tag
    python Code/visualize_results.py --config HEAVY --scenario S2_9R_9S

    # custom directories
    python Code/visualize_results.py --out-dir my_runs --fig-dir my_runs/figures
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style ──────────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

# Canonical method colours — notebook palette extended with DoGE / Skill-It.
METHOD_PALETTE = {
    "Aioli":              "#2ca02c",
    "Stratified":         "#7f7f7f",
    "DoReMi-style":       "#9467bd",
    "Group-DRO":          "#8c564b",
    "DoGE":               "#ff7f0e",
    "Skill-It":           "#1f77b4",
    "Gen-Delta":          "#e377c2",
    "Improvement-Window": "#bcbd22",
}

METHOD_ORDER = [
    "Aioli", "Stratified", "DoReMi-style", "Group-DRO",
    "DoGE", "Skill-It", "Gen-Delta", "Improvement-Window",
]

ALL_CONFIGS   = ["SUPER_LIGHT", "LIGHT", "HEAVY"]
ALL_SCENARIOS = ["S1_1R_9S", "S2_9R_9S", "S3_9R_1S"]


# ── helpers ────────────────────────────────────────────────────────────────

def _c(method: str) -> str:
    return METHOD_PALETTE.get(method, "#333333")


def _palette(n: int):
    if _HAS_SNS:
        return sns.color_palette("husl", n)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def _ordered_methods(df: pd.DataFrame, col: str = "method") -> list:
    present = set(df[col].unique())
    return [m for m in METHOD_ORDER if m in present]


# ── figure 1 – comparative synthetic share ─────────────────────────────────

def fig_synth_share(df: pd.DataFrame, tag: str, fig_dir: Path):
    """All methods on one axes: total synthetic share per round, ±1σ across seeds."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in _ordered_methods(df):
        sub = df[df["method"] == method].sort_values("round")
        ax.plot(sub["round"], sub["synth_share_mean"],
                color=_c(method), marker="o", markersize=3, label=method)
        if sub["synth_share_std"].gt(0).any():
            ax.fill_between(sub["round"],
                            sub["synth_share_mean"] - sub["synth_share_std"],
                            sub["synth_share_mean"] + sub["synth_share_std"],
                            color=_c(method), alpha=0.15)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Round")
    ax.set_ylabel("Synthetic data share  (Σ p over synthetic domains)")
    ax.set_title(f"Synthetic share vs. round  [{tag}]")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, fig_dir / f"{tag}_comparative_synth_share.png")


# ── figure 2 – L2 divergence from uniform ─────────────────────────────────

def fig_divergence(df: pd.DataFrame, tag: str, fig_dir: Path):
    """L2 distance from the uniform mixture per round, all methods, ±1σ."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in _ordered_methods(df):
        sub = df[df["method"] == method].sort_values("round")
        ax.plot(sub["round"], sub["l2_mean"],
                color=_c(method), marker="d", markersize=3, label=method)
        if sub["l2_std"].gt(0).any():
            ax.fill_between(sub["round"],
                            sub["l2_mean"] - sub["l2_std"],
                            sub["l2_mean"] + sub["l2_std"],
                            color=_c(method), alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("L₂ distance from uniform")
    ax.set_title(f"Divergence from uniform  [{tag}]")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, fig_dir / f"{tag}_divergence.png")


# ── figure 3 – final per-domain weights (horizontal grouped bar) ──────────

def fig_final_weights(df: pd.DataFrame, tag: str, fig_dir: Path):
    """3-column grouped bar chart: final weight per domain per method, ±1σ.

    Domains are distributed across three panels (6 per column for 18 domains) so
    the figure is roughly square rather than a tall strip.  All panels share the
    same x-axis scale for visual comparability.  A dashed rule marks the
    real-to-synthetic boundary within any column where the split falls.
    """
    from matplotlib.patches import Patch

    methods = _ordered_methods(df)
    if not methods:
        return

    # Sort: real (alphabetical) then synthetic (alphabetical)
    domains_df = (df[["domain", "is_synth"]].drop_duplicates()
                  .sort_values(["is_synth", "domain"]))
    domains = domains_df["domain"].tolist()
    synth_flag = {row["domain"]: row["is_synth"]
                  for _, row in domains_df.iterrows()}
    n_dom, n_meth = len(domains), len(methods)

    n_cols = 3
    n_per_col = (n_dom + n_cols - 1) // n_cols  # ceiling division

    bar_h = min(0.80 / n_meth, 0.26)
    panel_h = n_per_col * (bar_h * n_meth + 0.18)
    fig_h = max(5.0, panel_h * 1.5)
    fig_w = 5.5 * n_cols  # ~16.5 inches wide → roughly square for 6-domain panels

    # Determine global x-axis maximum for a shared scale
    x_max = 0.0
    for method in methods:
        sub = df[df["method"] == method]
        x_max = max(x_max, (sub["weight_mean"] + sub["weight_std"]).max())
    x_max = x_max * 1.08  # small margin

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), sharey=False)

    for col_idx, ax in enumerate(axes):
        col_domains = domains[col_idx * n_per_col: (col_idx + 1) * n_per_col]
        n_d = len(col_domains)

        for mi, method in enumerate(methods):
            sub = df[df["method"] == method].set_index("domain")
            offsets = [i + (mi - n_meth / 2 + 0.5) * bar_h for i in range(n_d)]
            ws = [float(sub.loc[d, "weight_mean"]) if d in sub.index else 0.0
                  for d in col_domains]
            es = [float(sub.loc[d, "weight_std"])  if d in sub.index else 0.0
                  for d in col_domains]
            ax.barh(offsets, ws, height=bar_h * 0.88, xerr=es, label=method,
                    color=_c(method), capsize=2,
                    error_kw={"linewidth": 0.8}, alpha=0.85)

        tick_labels = [d + (" [S]" if synth_flag.get(d) else "")
                       for d in col_domains]
        ax.set_yticks(range(n_d))
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Final weight", fontsize=9)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(1 / n_dom, color="grey", lw=0.8, ls=":")  # uniform baseline

        # Dashed rule between real and synthetic domains (if split falls in this column)
        for i in range(1, n_d):
            if synth_flag.get(col_domains[i]) and not synth_flag.get(col_domains[i - 1]):
                ax.axhline(i - 0.5, color="black", lw=1.0, ls="--", alpha=0.4)

    # Shared legend on the right side of the last panel
    legend_handles = [Patch(facecolor=_c(m), alpha=0.85, label=m) for m in methods]
    axes[-1].legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
                    loc="upper left", fontsize=8)

    fig.tight_layout(w_pad=1.5)
    _save(fig, fig_dir / f"{tag}_final_weights.png")


# ── figure 4 – per-method weight evolution ────────────────────────────────

def fig_weight_evolution(df: pd.DataFrame, tag: str, fig_dir: Path):
    """One figure per method: per-domain weight trajectories (solid=real, dashed=synth), ±1σ."""
    if df.empty:
        print("  [skip] weight_trajectories.csv not found — re-run run_experiment.py to generate it")
        return

    grp = (df.groupby(["method", "round", "pretty", "is_synth"])["weight"]
             .agg(w_mean="mean", w_std="std").reset_index())

    for method in _ordered_methods(grp, col="method"):
        sub = grp[grp["method"] == method]
        domain_list = sorted(sub["pretty"].unique())
        colors = _palette(len(domain_list))

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, domain in enumerate(domain_list):
            d = sub[sub["pretty"] == domain].sort_values("round")
            is_synth = bool(d["is_synth"].iloc[0])
            ls = "--" if is_synth else "-"
            label = domain + (" [S]" if is_synth else "")
            ax.plot(d["round"], d["w_mean"], ls=ls, color=colors[i],
                    label=label, lw=2.0, marker="o", markersize=4)
            has_band = d["w_std"].notna().any() and d["w_std"].gt(0).any()
            if has_band:
                ax.fill_between(d["round"],
                                d["w_mean"] - d["w_std"],
                                d["w_mean"] + d["w_std"],
                                color=colors[i], alpha=0.12)

        ax.set_xlabel("Round")
        ax.set_ylabel("Weight  (p)")
        ax.set_title(f"Weight evolution: {method}\n"
                     f"(solid = real domains, dashed = synthetic)")
        ncol = 1 if len(domain_list) < 10 else 2
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=ncol, fontsize=8)
        fig.tight_layout()
        safe = method.replace("/", "_").replace(" ", "_")
        _save(fig, fig_dir / f"{tag}_{safe}_weight_evolution.png")


# ── figure 5 – validation-loss trajectories ───────────────────────────────

def fig_val_loss(df: pd.DataFrame, tag: str, fig_dir: Path):
    """Per-domain validation-loss curves per method that logged val_losses (typically Aioli).
    Solid lines = real domains, dashed = synthetic. ±1σ bands across seeds."""
    if df.empty:
        print("  [skip] val_loss_trajectories.csv not found — re-run run_experiment.py to generate it")
        return

    grp = (df.groupby(["method", "round", "pretty", "is_synth"])["val_loss"]
             .agg(l_mean="mean", l_std="std").reset_index())

    for method in _ordered_methods(grp, col="method"):
        sub = grp[grp["method"] == method]
        domain_list = sorted(sub["pretty"].unique())
        colors = _palette(len(domain_list))

        fig, ax = plt.subplots(figsize=(12, 7))
        for i, domain in enumerate(domain_list):
            d = sub[sub["pretty"] == domain].sort_values("round")
            is_synth = bool(d["is_synth"].iloc[0])
            ls, lw, alpha = ("--", 2.5, 1.0) if is_synth else ("-", 1.8, 0.65)
            label = domain + (" [S]" if is_synth else "")
            ax.plot(d["round"], d["l_mean"], ls=ls, lw=lw, alpha=alpha,
                    color=colors[i], label=label, marker="o", markersize=3)
            has_band = d["l_std"].notna().any() and d["l_std"].gt(0).any()
            if has_band:
                ax.fill_between(d["round"],
                                d["l_mean"] - d["l_std"],
                                d["l_mean"] + d["l_std"],
                                color=colors[i], alpha=0.10)

        ax.set_xlabel("Round")
        ax.set_ylabel("Validation loss")
        ax.set_title(f"Validation-loss trajectories: {method}  [{tag}]\n"
                     f"(solid = real, dashed = synthetic)")
        ncol = 1 if len(domain_list) < 10 else 2
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=ncol, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = method.replace("/", "_").replace(" ", "_")
        _save(fig, fig_dir / f"{tag}_{safe}_val_loss.png")


# ── figure 6 – held-out test loss ─────────────────────────────────────────

def fig_test_loss(df: pd.DataFrame, tag: str, fig_dir: Path):
    """Three-panel horizontal bar chart: overall / real / synth test loss per method."""
    if df.empty:
        return
    sub = df[df["eval"] == "test_loss"]
    if sub.empty:
        return

    methods = _ordered_methods(sub)
    sub = sub.set_index("method")

    cols   = ["test_loss_mean", "real_mean", "synth_mean"]
    titles = ["Overall Test Loss", "Real Domains", "Synthetic Domains"]

    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, len(methods) * 0.55 + 1.5)), sharey=True)
    for ax, col, title in zip(axes, cols, titles):
        vals   = [float(sub.loc[m, col])             if m in sub.index else float("nan") for m in methods]
        errs   = [float(sub.loc[m, "test_loss_std"]) if m in sub.index else 0.0          for m in methods]
        colors = [_c(m) for m in methods]
        ax.barh(methods, vals,
                xerr=(errs if col == "test_loss_mean" else None),
                color=colors, capsize=3, error_kw={"linewidth": 0.8}, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cross-entropy loss (nats)")
        ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, fig_dir / f"{tag}_test_loss.png")


# ── cross-tier figures (HEAVY vs LIGHT vs SUPER_LIGHT) ────────────────────

TIER_ORDER = ["SUPER_LIGHT", "LIGHT", "HEAVY"]
TIER_FLOPS = {"SUPER_LIGHT": 2.5e14, "LIGHT": 1.2e16, "HEAVY": 2.3e17}
TIER_PRETTY = {"SUPER_LIGHT": "SUPER-LIGHT", "LIGHT": "LIGHT", "HEAVY": "HEAVY"}


def _flops_mathtext(f: float) -> str:
    exp = int(np.floor(np.log10(f)))
    mant = f / 10 ** exp
    return rf"${mant:.1f}{{\times}}10^{{{exp}}}$"


def _final_per_method(df: pd.DataFrame) -> dict:
    """method -> (share_mean, share_std, l2_mean, l2_std) at the final round."""
    out = {}
    for m in df["method"].unique():
        r = df[df["method"] == m].sort_values("round").iloc[-1]
        out[m] = (float(r["synth_share_mean"]), float(r["synth_share_std"]),
                  float(r["l2_mean"]), float(r["l2_std"]))
    return out


def fig_tier_share_panels(tier_dfs: dict, fig_dir: Path):
    """1x3 panels (SUPER-LIGHT | LIGHT | HEAVY): synthetic-share trajectories,
    all methods, +/-1 sigma over seeds. Shows the loss-monotone split emerge as
    the compute budget grows."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, cfg in zip(axes, TIER_ORDER):
        df = tier_dfs[cfg]
        for method in _ordered_methods(df):
            sub = df[df["method"] == method].sort_values("round")
            ax.plot(sub["round"], sub["synth_share_mean"], color=_c(method),
                    marker="o", markersize=3.5, label=method)
            if sub["synth_share_std"].gt(0).any():
                ax.fill_between(sub["round"],
                                sub["synth_share_mean"] - sub["synth_share_std"],
                                sub["synth_share_mean"] + sub["synth_share_std"],
                                color=_c(method), alpha=0.13)
        ax.axhline(0.5, color="grey", lw=0.9, ls=":")
        ax.set_title(f"{TIER_PRETTY[cfg]}\n{_flops_mathtext(TIER_FLOPS[cfg])} FLOPs/run",
                     fontsize=12)
        ax.set_xlabel("Reweighting round")
        ax.set_ylim(0.0, 1.02)
        ax.margins(x=0.02)
    axes[0].set_ylabel(r"Synthetic share  ($\sum p$ over synthetic)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, fig_dir / "f_tier_share_panels.png")


def fig_scale_to_signal(tier_dfs: dict, fig_dir: Path):
    """1x2: (a) final synthetic share vs compute (the scale-to-signal curve),
    (b) seed std of the final share vs compute (the noise floor). x on log FLOPs."""
    methods = _ordered_methods(tier_dfs["HEAVY"])
    xs = [TIER_FLOPS[c] for c in TIER_ORDER]
    finals = {c: _final_per_method(tier_dfs[c]) for c in TIER_ORDER}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) final share vs compute, +/-1 sigma
    for m in methods:
        ys = [finals[c][m][0] for c in TIER_ORDER]
        es = [finals[c][m][1] for c in TIER_ORDER]
        axa.errorbar(xs, ys, yerr=es, color=_c(m), marker="o", markersize=6,
                     capsize=4, lw=2.0, label=m)
    axa.axhline(0.5, color="grey", lw=0.9, ls=":")
    axa.set_xscale("log")
    axa.set_xticks(xs)
    axa.set_xticklabels([f"{TIER_PRETTY[c]}\n{_flops_mathtext(TIER_FLOPS[c])}" for c in TIER_ORDER],
                        fontsize=9)
    axa.set_ylabel("Final synthetic share")
    axa.set_xlabel("Compute per run (FLOPs)")
    axa.set_title("(a) Signal: where the split appears")
    axa.set_ylim(0.0, 1.0)
    axa.legend(ncol=2, fontsize=9, loc="upper right")

    # (b) seed std of final share vs compute (noise floor)
    for m in methods:
        ys = [finals[c][m][1] for c in TIER_ORDER]
        axb.plot(xs, ys, color=_c(m), marker="s", markersize=6, lw=2.0, label=m)
    axb.set_xscale("log")
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{TIER_PRETTY[c]}\n{_flops_mathtext(TIER_FLOPS[c])}" for c in TIER_ORDER],
                        fontsize=9)
    axb.set_ylabel("Seed std of final synthetic share")
    axb.set_xlabel("Compute per run (FLOPs)")
    axb.set_title("(b) Noise: seed spread collapses with compute")
    fig.tight_layout()
    _save(fig, fig_dir / "f_scale_to_signal.png")


def make_cross_tier_figures(out_dir: Path, fig_dir: Path, scenario: str = "S2_9R_9S"):
    tier_dfs = {}
    for cfg in TIER_ORDER:
        p = out_dir / f"agg_{cfg}_{scenario}_synth_share.csv"
        if not p.exists():
            print(f"  [skip cross-tier] {p.name} not found")
            return
        tier_dfs[cfg] = pd.read_csv(p)
    print("\n-- cross-tier figures (HEAVY vs LIGHT vs SUPER_LIGHT) --")
    fig_tier_share_panels(tier_dfs, fig_dir)
    fig_scale_to_signal(tier_dfs, fig_dir)


# ── dispatcher ────────────────────────────────────────────────────────────

def process_tag(tag: str, out_dir: Path, fig_dir: Path):
    print(f"\n-- {tag} --")

    # figures 1 & 2: synth share + divergence
    p = out_dir / f"agg_{tag}_synth_share.csv"
    if p.exists():
        df_ss = pd.read_csv(p)
        fig_synth_share(df_ss, tag, fig_dir)
        fig_divergence(df_ss, tag, fig_dir)
    else:
        print(f"  [skip] {p.name} not found")

    # figure 3: final weights
    p = out_dir / f"agg_{tag}_final_weights.csv"
    if p.exists():
        fig_final_weights(pd.read_csv(p), tag, fig_dir)
    else:
        print(f"  [skip] {p.name} not found")

    # figure 4: per-method weight evolution
    p = out_dir / f"agg_{tag}_weight_trajectories.csv"
    fig_weight_evolution(pd.read_csv(p) if p.exists() else pd.DataFrame(), tag, fig_dir)

    # figure 5: val-loss trajectories
    p = out_dir / f"agg_{tag}_val_loss_trajectories.csv"
    fig_val_loss(pd.read_csv(p) if p.exists() else pd.DataFrame(), tag, fig_dir)

    # figure 6: test-loss bar chart (optional)
    p = out_dir / f"agg_{tag}_test_loss.csv"
    fig_test_loss(pd.read_csv(p) if p.exists() else pd.DataFrame(), tag, fig_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Generate all notebook figures from run_experiment.py CSV outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--out-dir",  default=str(REPO_ROOT / "Results" / "main"),
                    help="Directory containing CSVs from run_experiment.py  (default: Results/main)")
    ap.add_argument("--fig-dir",  default=str(REPO_ROOT / "Results" / "figures"),
                    help="Output directory for PNG figures  (default: Results/figures)")
    ap.add_argument("--config",   default="ALL",
                    help="Config name: SUPER_LIGHT, LIGHT, HEAVY, or ALL  (default: ALL)")
    ap.add_argument("--scenario", default="ALL",
                    help="Scenario: S1_1R_9S, S2_9R_9S, S3_9R_1S, or ALL  (default: ALL)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs   = ALL_CONFIGS   if args.config   == "ALL" else [c.strip() for c in args.config.split(",")]
    scenarios = ALL_SCENARIOS if args.scenario == "ALL" else [s.strip() for s in args.scenario.split(",")]

    found = False
    for cfg in configs:
        for scen in scenarios:
            tag = f"{cfg}_{scen}"
            if not (out_dir / f"agg_{tag}_synth_share.csv").exists():
                continue
            process_tag(tag, out_dir, fig_dir)
            found = True

    if not found:
        print(f"\nNo CSV files found in {out_dir}/ for "
              f"configs={configs} × scenarios={scenarios}.")
        print("Run run_experiment.py first to generate the data.")
        sys.exit(1)

    # Cross-tier comparison figures (HEAVY vs LIGHT vs SUPER_LIGHT) for S2.
    for scen in scenarios:
        if all((out_dir / f"agg_{c}_{scen}_synth_share.csv").exists() for c in TIER_ORDER):
            make_cross_tier_figures(out_dir, fig_dir, scen)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
