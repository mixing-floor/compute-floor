"""DoGE execution with direct raw-score and provenance-contribution logging."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _validate_provenance_order(domains: list[dict[str, Any]]) -> int:
    flags = [bool(domain["is_synth"]) for domain in domains]
    n_real = flags.count(False)
    if n_real == 0 or n_real == len(flags):
        raise ValueError(
            "DoGE provenance diagnostics require both real and synthetic domains."
        )
    if flags != [False] * n_real + [True] * (len(flags) - n_real):
        raise ValueError(
            "Expected real domains followed by synthetic domains in the scenario."
        )
    return n_real


def gradient_alignment_diagnostics(
    core: Any,
    model: Any,
    tokenizer: Any,
    domains: list[dict[str, Any]],
    device: str,
    est_batches: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Compute DoGE scores and their real/synthetic target-gradient components.

    The batches and total score match the submitted two-pass estimator. During pass 1,
    the cumulative gradient is snapshotted after the real domains. This adds logging
    without drawing an extra batch or changing the training-data cursor/RNG stream.
    """
    n_real = _validate_provenance_order(domains)
    core.attach_train_cursors(domains)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.train()

    def domain_backward(domain_index: int) -> None:
        for _ in range(est_batches):
            one_hot = np.eye(len(domains))[domain_index]
            picks = core.mixture_indices_per_batch(domains, one_hot, batch_size)
            if not picks:
                continue
            batch = core.batchify(tokenizer, domains, picks, device)
            loss = model(**batch).loss / est_batches
            loss.backward()

    model.zero_grad(set_to_none=True)
    gradient_real = None
    for domain_index in range(len(domains)):
        domain_backward(domain_index)
        if domain_index == n_real - 1:
            gradient_real = [
                parameter.grad.detach().float().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter, dtype=torch.float32)
                for parameter in params
            ]
    if gradient_real is None:
        raise RuntimeError("Failed to snapshot the aggregate real-domain gradient.")
    gradient_total = [
        parameter.grad.detach().float().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in params
    ]

    scores = np.zeros(len(domains), dtype=float)
    alignment_real = np.zeros(len(domains), dtype=float)
    for domain_index in range(len(domains)):
        model.zero_grad(set_to_none=True)
        domain_backward(domain_index)
        dot_total = 0.0
        dot_real = 0.0
        for parameter, total_gradient, real_gradient in zip(
            params, gradient_total, gradient_real
        ):
            if parameter.grad is None:
                continue
            current = parameter.grad.detach().float()
            dot_total += torch.sum(current * total_gradient).item()
            dot_real += torch.sum(current * real_gradient).item()
        scores[domain_index] = dot_total
        alignment_real[domain_index] = dot_real

    model.zero_grad(set_to_none=True)
    model.eval()
    alignment_synthetic = scores - alignment_real
    del gradient_total, gradient_real
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError(f"Non-finite DoGE score detected: {scores}")
    return {
        "raw_scores": scores,
        "alignment_to_real": alignment_real,
        "alignment_to_synthetic": alignment_synthetic,
    }


def run_doge_logged(
    core: Any,
    model: Any,
    tokenizer: Any,
    domains: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    steps: int,
    mu: float = 0.2,
) -> dict[str, list[Any]]:
    """Run the submitted DoGE update while retaining every signal used by EG."""
    from tqdm.auto import tqdm

    n_domains = len(domains)
    weights = np.ones(n_domains) / n_domains
    history = {
        "round": [],
        "p": [],
        "signal_round": [],
        "signal_step": [],
        "raw_scores": [],
        "normalized_scores": [],
        "score_normalizer": [],
        "alignment_to_real": [],
        "alignment_to_synthetic": [],
        "pre_update_p": [],
        "instantaneous_p": [],
    }
    trainer = core.CompactAioli(
        model, tokenizer, domains, core.AioliArgs(0, 0, 0, 0, 1.0), cfg, device
    )
    est_batches = cfg.get("doge_est_batches", 2)
    averaged_weights = np.zeros(n_domains)
    average_count = 0
    step = 0
    round_index = 0
    progress = tqdm(total=steps, desc="DoGE (logged)")
    while step < steps:
        diagnostics = gradient_alignment_diagnostics(
            core, model, tokenizer, domains, device, est_batches, cfg["batch_size"]
        )
        raw_scores = diagnostics["raw_scores"]
        normalizer = float(np.abs(raw_scores).mean() + 1e-8)
        normalized_scores = raw_scores / normalizer
        pre_update = weights.copy()
        weights = weights * np.exp(mu * normalized_scores)
        weights = weights / weights.sum()

        averaged_weights += weights
        average_count += 1
        reported_weights = averaged_weights / average_count
        history["round"].append(round_index)
        history["p"].append(reported_weights.copy())
        history["signal_round"].append(round_index)
        history["signal_step"].append(step)
        history["raw_scores"].append(raw_scores.copy())
        history["normalized_scores"].append(normalized_scores.copy())
        history["score_normalizer"].append(normalizer)
        history["alignment_to_real"].append(diagnostics["alignment_to_real"].copy())
        history["alignment_to_synthetic"].append(
            diagnostics["alignment_to_synthetic"].copy()
        )
        history["pre_update_p"].append(pre_update)
        history["instantaneous_p"].append(weights.copy())

        run_steps = min(cfg["update_steps"], steps - step)
        if run_steps <= 0:
            break
        trainer._train_steps(run_steps, weights)
        step += run_steps
        progress.update(run_steps)
        round_index += 1
    progress.close()
    history["round"].append(round_index)
    history["p"].append((averaged_weights / max(average_count, 1)).copy())
    return history
