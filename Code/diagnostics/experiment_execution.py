"""Seed-level execution paths for the two diagnostic experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .doge_diagnostics import run_doge_logged
from .skillit_diagnostics import run_skillit_logged


@contextmanager
def logged_interaction_methods_patch(core: Any) -> Iterator[None]:
    original_doge = core.run_doge
    original_skillit = core.run_skillit

    def adapter(
        model: Any,
        tokenizer: Any,
        domains: Any,
        cfg: Any,
        device: str,
        steps: int,
        mu: float = 0.2,
    ):
        return run_doge_logged(core, model, tokenizer, domains, cfg, device, steps, mu)

    def skillit_adapter(
        model: Any,
        tokenizer: Any,
        domains: Any,
        cfg: Any,
        device: str,
        steps: int,
        eta: float = 0.2,
        window: int = 3,
    ):
        return run_skillit_logged(
            core, model, tokenizer, domains, cfg, device, steps, eta, window
        )

    core.run_doge = adapter
    core.run_skillit = skillit_adapter
    try:
        yield
    finally:
        core.run_doge = original_doge
        core.run_skillit = original_skillit


def run_doge_only_seed(
    core: Any,
    domains: list[dict[str, Any]],
    tokenizer: Any,
    cfg: dict[str, Any],
    device: str,
    seed: int,
    evaluate_test: bool,
) -> dict[str, Any]:
    core.set_all_seeds(seed)
    model = core.new_model(cfg, device)
    result = {
        "histories": {},
        "test_loss": {},
        "static_test_loss": {},
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    try:
        result["histories"]["DoGE"] = run_doge_logged(
            core, model, tokenizer, domains, cfg, device, cfg["max_steps"]
        )
        if evaluate_test:
            losses = core.eval_domain_losses(
                model,
                tokenizer,
                domains,
                device,
                "test",
                eval_bsz=cfg["batch_size"] * 2,
            )
            result["test_loss"]["DoGE"] = losses.tolist()
    finally:
        del model
        core._free()
    return result


def run_full_seed(
    core: Any,
    domains: list[dict[str, Any]],
    tokenizer: Any,
    cfg: dict[str, Any],
    device: str,
    seed: int,
    evaluate_test: bool,
) -> dict[str, Any]:
    with logged_interaction_methods_patch(core):
        return core.run_scenario_one_seed(
            domains,
            tokenizer,
            cfg,
            device,
            seed,
            run_baselines=True,
            eval_test=evaluate_test,
            static_eval=False,
            extra_heuristics=False,
        )
