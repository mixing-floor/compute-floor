"""CPU-safe behavioral test for Skill-It logging."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from diagnostics.skillit_diagnostics import run_skillit_logged
from diagnostics.skillit_tables import skillit_graph_table, skillit_signal_tables


class FakeProgress:
    def __init__(self, *args, **kwargs):
        return None

    def update(self, steps):
        return None

    def close(self):
        return None


if "tqdm.auto" not in sys.modules:
    tqdm_package = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = FakeProgress
    tqdm_package.auto = tqdm_auto
    sys.modules.update({"tqdm": tqdm_package, "tqdm.auto": tqdm_auto})


class FakeTrainer:
    def __init__(self, model, tokenizer, domains, args, cfg, device):
        self.losses = np.arange(1, len(domains) + 1, dtype=float)

    def _train_steps(self, steps, weights):
        return None

    def _eval_val(self):
        return self.losses.copy()


class FakeSkillItCore:
    CompactAioli = FakeTrainer

    @staticmethod
    def AioliArgs(*args):
        return args

    @staticmethod
    def learn_skill_graph(tokenizer, domains, cfg, device, probe_steps):
        return np.eye(len(domains))


class SkillItLoggingTests(unittest.TestCase):
    def test_graph_scores_and_tables_are_retained(self):
        domains = [{"is_synth": value} for value in (False, False, True, True)]
        cfg = {"update_steps": 1, "skillit_graph_probe_steps": 2}
        history = run_skillit_logged(
            FakeSkillItCore, object(), object(), domains, cfg, "cpu", steps=2
        )
        self.assertEqual(len(history["p"]), 3)
        np.testing.assert_allclose(history["skill_graph"], np.eye(4))
        np.testing.assert_allclose(history["raw_scores"][0], [1.0, 2.0, 3.0, 4.0])

        result = {"seed": 0, "histories": {"Skill-It": history}}
        pretty = ["r1", "r2", "s1", "s2"]
        flags = np.array([False, False, True, True])
        graph = skillit_graph_table([result], pretty, flags)
        signals, summary = skillit_signal_tables([result], pretty, flags)
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(signals), 8)
        self.assertGreater(
            summary.iloc[0]["synthetic_minus_real_normalized_score"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
