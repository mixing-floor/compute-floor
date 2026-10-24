"""Immutable definitions of the two rebuttal experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    output_slug: str
    model_name: str
    run_all_methods: bool
    purpose: str
    rng_protocol: str


EXPERIMENTS = {
    "doge160_signal": ExperimentSpec(
        name="DoGE raw-signal audit on Pythia-160M",
        output_slug="doge160_signal_heavy_s2",
        model_name="EleutherAI/pythia-160m",
        run_all_methods=False,
        purpose="Directly log the gradient-alignment signal consumed by DoGE.",
        rng_protocol=(
            "Method-local replication: reset each seed immediately before DoGE model "
            "initialization; not the sequential RNG continuation of the submitted all-method job."
        ),
    ),
    "pythia70m_heavy": ExperimentSpec(
        name="Matched Pythia-70M replication at HEAVY settings",
        output_slug="pythia70m_heavy_s2",
        model_name="EleutherAI/pythia-70m",
        run_all_methods=True,
        purpose=(
            "Test whether the qualitative family split repeats at a second model size; "
            "only model_name changes in the configured protocol."
        ),
        rng_protocol="Submitted protocol: reset once per seed, then run all methods sequentially.",
    ),
}


def build_config(core: Any, spec: ExperimentSpec) -> dict[str, Any]:
    """Return HEAVY with only the requested checkpoint override."""
    cfg = dict(**core.COMMON, **core.CONFIGS["HEAVY"])
    cfg["model_name"] = spec.model_name
    return cfg


def changed_heavy_fields(
    core: Any, cfg: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """List changes relative to the submitted HEAVY configuration."""
    heavy = dict(**core.COMMON, **core.CONFIGS["HEAVY"])
    keys = sorted(set(heavy) | set(cfg))
    return {
        key: (heavy.get(key), cfg.get(key))
        for key in keys
        if heavy.get(key) != cfg.get(key)
    }


def validate_spec_config(
    core: Any, spec: ExperimentSpec, cfg: Mapping[str, Any]
) -> None:
    changes = changed_heavy_fields(core, cfg)
    expected = (
        {}
        if spec.model_name == core.CONFIGS["HEAVY"]["model_name"]
        else {"model_name": (core.CONFIGS["HEAVY"]["model_name"], spec.model_name)}
    )
    if changes != expected:
        raise ValueError(
            f"Unexpected changes relative to HEAVY: {changes}; expected {expected}"
        )
