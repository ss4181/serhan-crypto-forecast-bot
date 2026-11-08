from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from oi_coverage_audit import BAR, align_oi_features


class OIAlignmentTests(unittest.TestCase):
    def source(self):
        return pd.DataFrame({"timestamp_ms": np.arange(30) * BAR,
                             "open_interest": np.arange(30) + 100.,
                             "open_interest_value": np.arange(30) * 1000. + 100000.})

    def test_strict_past_and_future_invariance(self):
        oi = self.source()
        closes = np.array([15 * BAR - 1])
        before = align_oi_features(closes, oi)
        self.assertEqual(int(before.oi_source_ms.iloc[0]), 13 * BAR)
        self.assertAlmostEqual(before.oi_1h_change.iloc[0], 113 / 101 - 1)
        oi.loc[oi.timestamp_ms >= 14 * BAR, "open_interest"] = 1e9
        after = align_oi_features(closes, oi)
        pd.testing.assert_frame_equal(before, after)

    def test_gap_is_unknown_not_forward_filled(self):
        oi = self.source().drop(index=7)
        result = align_oi_features(np.array([15 * BAR - 1]), oi)
        self.assertFalse(result.oi_window_complete.iloc[0])
        self.assertTrue(np.isnan(result.oi_1h_change.iloc[0]))

    def test_price_valuation_does_not_fake_quantity_growth(self):
        oi = self.source()
        oi["open_interest"] = 100.
        result = align_oi_features(np.array([15 * BAR - 1]), oi)
        self.assertEqual(result.oi_1h_change.iloc[0], 0.)

    def test_duplicate_rejected_and_zero_lag_rejected(self):
        oi = self.source()
        with self.assertRaises(ValueError):
            align_oi_features(np.array([15 * BAR - 1]), pd.concat([oi, oi.iloc[[1]]]))
        with self.assertRaises(ValueError):
            align_oi_features(np.array([15 * BAR - 1]), oi, lag_bars=0)


if __name__ == "__main__":
    unittest.main()
