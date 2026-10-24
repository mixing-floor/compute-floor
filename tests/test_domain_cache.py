"""CPU-safe integrity test for the shared token-block cache."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from diagnostics.domain_cache import load_or_build_domains


class FakeShard:
    def __init__(self, arrays):
        self.arrays = arrays


class FakeCacheCore:
    __file__ = __file__
    SCENARIOS = {"S2": {"real": ["r"], "synth": ["s"]}}
    REAL_DOMAINS = {"r": ("real/source", "train", "Real")}
    SYN_DOMAINS = {"s": ("synthetic/source", "train", "Synthetic")}
    DomainShard = FakeShard

    @staticmethod
    def build_scenario_domains(tokenizer, loader, scenario, cfg, split_seed):
        domains = []
        for name, pretty, is_synth, offset in (
            ("r", "Real", False, 0),
            ("s", "Synthetic", True, 10),
        ):
            shards = {
                split: FakeShard(np.arange(8, dtype=np.int32).reshape(2, 4) + offset)
                for split in ("train", "val", "test")
            }
            domains.append(
                {**shards, "name": name, "pretty": pretty, "is_synth": is_synth}
            )
        return domains


class DomainCacheTests(unittest.TestCase):
    def test_cache_reuse_and_checksum_guard(self):
        cfg = {
            "tokenizer_name": "tokenizer",
            "context_length": 4,
            "train_tokens_per_domain": 8,
            "val_tokens_per_domain": 8,
            "test_tokens_per_domain": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domains, cache_dir, built = load_or_build_domains(
                FakeCacheCore, None, None, cfg, "S2", 1234, root
            )
            self.assertTrue(built)
            self.assertEqual(len(domains), 2)
            _, reused_dir, rebuilt = load_or_build_domains(
                FakeCacheCore, None, None, cfg, "S2", 1234, root
            )
            self.assertFalse(rebuilt)
            self.assertEqual(cache_dir, reused_dir)

            target = cache_dir / "domain_00_train.npy"
            target.write_bytes(target.read_bytes() + b"corruption")
            with self.assertRaises(RuntimeError):
                load_or_build_domains(FakeCacheCore, None, None, cfg, "S2", 1234, root)


if __name__ == "__main__":
    unittest.main()
