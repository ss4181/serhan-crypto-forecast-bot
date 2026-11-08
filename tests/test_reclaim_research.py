from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from reclaim_study_20261022 import signal_indices


class ReclaimResearchTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({"open": np.full(400, 100.), "high": np.full(400, 101.),
                              "low": np.full(400, 99.), "close": np.full(400, 100.),
                              "volume": np.full(400, 100.), "quote_volume": np.full(400, 50_000.)})
        frame.loc[300, ["open", "high", "low", "close", "volume"]] = [100., 100., 95., 96., 250.]
        frame.loc[301, ["open", "high", "low", "close"]] = [96., 102., 96., 101.]
        return frame

    def test_signal_has_closed_reclaim_and_liquidity(self):
        frame = self.frame()
        self.assertIn(301, signal_indices(frame))
        frame.loc[301, "close"] = 99.
        self.assertNotIn(301, signal_indices(frame))
        frame = self.frame()
        frame["quote_volume"] = 1.
        self.assertEqual(len(signal_indices(frame)), 0)

    def test_future_mutation_and_prefix_invariance(self):
        frame = self.frame()
        expected = signal_indices(frame)
        self.assertIn(301, expected)  # non-vacuous invariance test
        prefix = signal_indices(frame.iloc[:302])
        np.testing.assert_array_equal(prefix, expected[expected < 302])
        frame.loc[302:, ["open", "high", "low", "close", "volume", "quote_volume"]] = 1e9
        changed = signal_indices(frame)
        np.testing.assert_array_equal(expected[expected < 302], changed[changed < 302])

    def test_fall_bar_cannot_set_its_own_reference_volume(self):
        frame = self.frame()
        frame.loc[300, "volume"] = 199.
        self.assertNotIn(301, signal_indices(frame))


if __name__ == "__main__":
    unittest.main()
