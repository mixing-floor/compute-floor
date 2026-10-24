"""CPU-safe unit tests for the diagnostics package."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from diagnostics.analysis_tables import doge_signal_tables, l2_decomposition
from diagnostics.artifact_io import write_final_artifacts
from diagnostics.experiment_specs import (
    EXPERIMENTS,
    build_config,
    changed_heavy_fields,
    validate_spec_config,
)
from diagnostics.seed_storage import load_seed_result, save_seed_result


class FakeConfigCore:
    COMMON = {
        "tokenizer_name": "EleutherAI/pythia-160m",
        "weight_decay": 0.0,
        "lr": 5e-4,
        "save_dir": "./aioli_runs",
    }
    CONFIGS = {
        "HEAVY": {
            "model_name": "EleutherAI/pythia-160m",
            "context_length": 2048,
            "batch_size": 8,
            "max_steps": 14400,
            "update_steps": 2200,
            "lp_rounds": 6,
            "lp_steps": 10,
            "warmup_steps": 500,
            "eta": 0.2,
            "one_hot_factor": 0.625,
            "aioli_normalize_A": True,
            "ema": 0.1,
            "train_tokens_per_domain": 1_000_000,
            "val_tokens_per_domain": 100_000,
            "test_tokens_per_domain": 100_000,
            "doge_est_batches": 2,
            "skillit_graph_probe_steps": 200,
        }
    }


class FakeArtifactCore:
    @staticmethod
    def aggregate_over_seeds(per_seed, is_synth):
        return {
            "DoGE": {
                "rounds": np.array([0, 1]),
                "share_mean": np.array([0.5, 0.5]),
                "share_std": np.zeros(2),
                "l2_mean": np.zeros(2),
                "l2_std": np.zeros(2),
            }
        }

    @staticmethod
    def save_error_band_figures(aggregate, prefix):
        return None

    @staticmethod
    def flop_table(cfg, n_params, tag, k):
        frame = pd.DataFrame(
            [
                {
                    "config": tag,
                    "method": "base (per run)",
                    "params": n_params,
                    "flops": 1,
                },
                {
                    "config": tag,
                    "method": "DoGE (total)",
                    "params": n_params,
                    "flops": 2,
                },
                {
                    "config": tag,
                    "method": "Skill-It (total)",
                    "params": n_params,
                    "flops": 3,
                },
            ]
        )
        return frame, 1


class DiagnosticsTests(unittest.TestCase):
    def test_70m_ablation_changes_only_model(self):
        spec = EXPERIMENTS["pythia70m_heavy"]
        cfg = build_config(FakeConfigCore, spec)
        validate_spec_config(FakeConfigCore, spec, cfg)
        self.assertEqual(
            changed_heavy_fields(FakeConfigCore, cfg),
            {"model_name": ("EleutherAI/pythia-160m", "EleutherAI/pythia-70m")},
        )
        self.assertEqual(cfg["tokenizer_name"], "EleutherAI/pythia-160m")

    def test_l2_decomposition_is_exact(self):
        weights = np.array([0.4, 0.1, 0.2, 0.3])
        values = l2_decomposition(weights, np.array([False, False, True, True]))
        self.assertAlmostEqual(
            values["l2_sq"],
            values["between_provenance_l2_sq"] + values["within_provenance_l2_sq"],
        )

    def test_doge_tables_report_class_gap(self):
        history = {
            "signal_round": [0],
            "signal_step": [0],
            "raw_scores": [np.array([1.0, 2.0, 3.0, 4.0])],
            "normalized_scores": [np.array([0.4, 0.8, 1.2, 1.6])],
            "score_normalizer": [2.5],
            "alignment_to_real": [np.array([0.2, 0.4, 0.6, 0.8])],
            "alignment_to_synthetic": [np.array([0.8, 1.6, 2.4, 3.2])],
            "pre_update_p": [np.full(4, 0.25)],
            "instantaneous_p": [np.array([0.1, 0.2, 0.3, 0.4])],
            "p": [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.3, 0.4])],
        }
        results = [{"seed": 0, "histories": {"DoGE": history}}]
        _, summary, blocks = doge_signal_tables(
            results, ["r1", "r2", "s1", "s2"], np.array([False, False, True, True])
        )
        self.assertAlmostEqual(
            summary.iloc[0]["synthetic_minus_real_normalized_score"], 0.8
        )
        self.assertEqual(len(blocks), 4)

    def test_seed_artifact_resume_metadata_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_0.npy"
            metadata = {"experiment": "test", "seed": 0}
            save_seed_result(path, {"seed": 0, "histories": {}}, metadata)
            loaded = load_seed_result(path, metadata)
            self.assertEqual(loaded["seed"], 0)
            with self.assertRaises(RuntimeError):
                load_seed_result(path, {"experiment": "different", "seed": 0})

    def test_final_artifact_bundle_is_analysis_ready(self):
        weights = np.full(4, 0.25)
        doge_history = {
            "round": [0, 1],
            "p": [weights, weights],
            "signal_round": [0],
            "signal_step": [0],
            "raw_scores": [np.array([1.0, 2.0, 3.0, 4.0])],
            "normalized_scores": [np.array([0.4, 0.8, 1.2, 1.6])],
            "score_normalizer": [2.5],
            "alignment_to_real": [np.array([0.2, 0.4, 0.6, 0.8])],
            "alignment_to_synthetic": [np.array([0.8, 1.6, 2.4, 3.2])],
            "pre_update_p": [weights],
            "instantaneous_p": [weights],
        }
        result = {
            "seed": 0,
            "histories": {"DoGE": doge_history},
            "n_params": 70,
            "test_loss": {"DoGE": [1.0, 2.0, 3.0, 4.0]},
        }
        cfg = build_config(FakeConfigCore, EXPERIMENTS["pythia70m_heavy"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            files = write_final_artifacts(
                FakeArtifactCore,
                output,
                "test",
                [result],
                ["r1", "r2", "s1", "s2"],
                [False, False, True, True],
                cfg,
                [0],
                "S2_9R_9S",
            )
            self.assertIn("doge_provenance_summary.csv", files)
            self.assertIn("doge_seed_summary.csv", files)
            self.assertIn("final_mixture_aggregate.csv", files)
            self.assertTrue((output / "test_loss_provenance_means.csv").exists())
            flops = pd.read_csv(output / "flops.csv")
            self.assertNotIn("Skill-It (total)", set(flops["method"]))


if __name__ == "__main__":
    unittest.main()
