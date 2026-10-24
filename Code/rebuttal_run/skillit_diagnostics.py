"""Skill-It execution with influence-graph and consumed-score logging."""

from __future__ import annotations

from typing import Any

import numpy as np


def run_skillit_logged(
    core: Any,
    model: Any,
    tokenizer: Any,
    domains: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    steps: int,
    eta: float = 0.2,
    window: int = 3,
) -> dict[str, Any]:
    """Run the submitted Skill-It update while retaining its graph and scores."""
    from tqdm.auto import tqdm

    n_domains = len(domains)
    graph = core.learn_skill_graph(
        tokenizer,
        domains,
        cfg,
        device,
        cfg.get("skillit_graph_probe_steps", 50),
    )
    weights = np.ones(n_domains) / n_domains
    history = {
        "round": [],
        "p": [],
        "skill_graph": graph.copy(),
        "signal_round": [],
        "signal_step": [],
        "val_losses": [],
        "raw_scores": [],
        "normalized_scores": [],
        "score_normalizer": [],
        "pre_update_p": [],
        "instantaneous_p": [],
    }
    trainer = core.CompactAioli(
        model, tokenizer, domains, core.AioliArgs(0, 0, 0, 0, 1.0), cfg, device
    )
    recent_scores = []
    step = 0
    round_index = 0
    progress = tqdm(total=steps, desc="Skill-It (logged)")
    while step < steps:
        history["round"].append(round_index)
        history["p"].append(weights.copy())
        run_steps = min(cfg["update_steps"], steps - step)
        if run_steps <= 0:
            break
        trainer._train_steps(run_steps, weights)
        step += run_steps
        progress.update(run_steps)

        validation_losses = np.nan_to_num(trainer._eval_val(), nan=0.0)
        recent_scores.append(graph @ validation_losses)
        if len(recent_scores) > window:
            recent_scores.pop(0)
        raw_scores = np.sum(recent_scores, axis=0)
        normalizer = float(np.abs(raw_scores).mean() + 1e-8)
        normalized_scores = raw_scores / normalizer
        pre_update = weights.copy()
        weights = weights * np.exp(eta * normalized_scores)
        weights = weights / weights.sum()

        history["signal_round"].append(round_index)
        history["signal_step"].append(step)
        history["val_losses"].append(validation_losses.copy())
        history["raw_scores"].append(raw_scores.copy())
        history["normalized_scores"].append(normalized_scores.copy())
        history["score_normalizer"].append(normalizer)
        history["pre_update_p"].append(pre_update)
        history["instantaneous_p"].append(weights.copy())
        round_index += 1
    progress.close()
    history["round"].append(round_index)
    history["p"].append(weights.copy())
    return history
