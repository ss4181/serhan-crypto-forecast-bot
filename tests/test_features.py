from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_forecaster.features import FEATURE_NAMES, build_supervised_dataset, compute_feature_frame


def sample_bars(count: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.002, count)
    close = 50_000 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    spread = close * rng.uniform(0.0005, 0.003, count)
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
    start = 1_700_000_000_000
    step = 300_000
    return pd.DataFrame(
        {
            "open_time_ms": start + np.arange(count) * step,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(8, 0.5, count),
            "close_time_ms": start + (np.arange(count) + 1) * step - 1,
        }
    )


class FeatureTests(unittest.TestCase):
    def test_features_do_not_change_when_future_is_appended(self) -> None:
        bars = sample_bars()
        prefix = compute_feature_frame(bars.iloc[:300])
        full = compute_feature_frame(bars)
        np.testing.assert_allclose(
            prefix.loc[250, FEATURE_NAMES].to_numpy(dtype=float),
            full.loc[250, FEATURE_NAMES].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )

    def test_dataset_uses_next_candle_and_drops_unlabelled_last_row(self) -> None:
        bars = sample_bars()
        dataset = build_supervised_dataset(bars)
        self.assertLess(int(dataset.close_time_ms[-1]), int(bars["close_time_ms"].iloc[-1]))
        self.assertEqual(dataset.x.shape[1], len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(dataset.x).all())
        self.assertTrue(set(np.unique(dataset.y)).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
