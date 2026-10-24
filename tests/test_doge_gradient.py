"""Optional tensor-level test for the DoGE provenance decomposition."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

try:
    import torch
    from diagnostics.doge_diagnostics import gradient_alignment_diagnostics

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    gradient_alignment_diagnostics = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ScalarGradientModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
            return types.SimpleNamespace(loss=self.weight * input_ids.float().sum())


class FakeGradientCore:
    @staticmethod
    def attach_train_cursors(domains):
        return None

    @staticmethod
    def mixture_indices_per_batch(domains, weights, batch_size):
        return [(int(np.argmax(weights)), 0)]

    @staticmethod
    def batchify(tokenizer, domains, picks, device):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch unavailable")
        coefficient = picks[0][0] + 1
        value = torch.tensor([[coefficient]], dtype=torch.float32, device=device)
        return {"input_ids": value, "labels": value}


class DogeGradientTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "A working PyTorch installation is required")
    def test_provenance_components_sum_to_total_score(self):
        domains = [{"is_synth": value} for value in (False, False, True, True)]
        diagnostics = gradient_alignment_diagnostics(
            FakeGradientCore(), ScalarGradientModel(), None, domains, "cpu", 1, 1
        )
        np.testing.assert_allclose(diagnostics["raw_scores"], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(
            diagnostics["alignment_to_real"], [3.0, 6.0, 9.0, 12.0]
        )
        np.testing.assert_allclose(
            diagnostics["alignment_to_real"] + diagnostics["alignment_to_synthetic"],
            diagnostics["raw_scores"],
        )


if __name__ == "__main__":
    unittest.main()
