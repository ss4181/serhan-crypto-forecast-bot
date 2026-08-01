from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from crypto_forecaster.features import FEATURE_NAMES, SupervisedDataset
from crypto_forecaster.model import (
    day_block_bootstrap_interval,
    fit_final_bundle,
    load_bundle,
    save_bundle,
    select_scenario,
    walk_forward_backtest,
    wilson_interval,
)


GATE = {
    "signal_threshold": 0.60,
    "minimum_signal_count": 100,
    "minimum_signal_accuracy": 0.53,
    "maximum_ece": 0.10,
    "round_trip_cost_bps": 10.0,
    "minimum_net_edge_bps": 0.0,
}


def predictive_dataset(count: int = 4000) -> SupervisedDataset:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(count, len(FEATURE_NAMES)))
    latent = 1.2 * x[:, 0] - 0.8 * x[:, 3] + 0.35 * x[:, 7]
    probabilities = 1 / (1 + np.exp(-latent))
    y = (rng.random(count) < probabilities).astype(float)
    returns = (2 * y - 1) * rng.uniform(0.1, 1.2, count) + rng.normal(0, 0.2, count)
    open_ms = 1_700_000_000_000 + np.arange(count) * 300_000
    return SupervisedDataset(
        x=x,
        y=y,
        open_time_ms=open_ms,
        close_time_ms=open_ms + 299_999,
        close=np.full(count, 50_000.0),
        atr=np.full(count, 500.0),
        atr_pct=np.exp(rng.normal(np.log(0.01), 0.2, count)),
        future_return_atr=returns,
        future_up_atr=np.maximum(returns, 0) + rng.uniform(0, 0.6, count),
        future_down_atr=np.maximum(-returns, 0) + rng.uniform(0, 0.6, count),
    )


class ModelTests(unittest.TestCase):
    def test_walk_forward_finds_only_chronological_predictive_signal(self) -> None:
        metrics = walk_forward_backtest(predictive_dataset(), **GATE)
        self.assertGreater(metrics.accuracy, 0.65)
        self.assertGreater(metrics.signal_accuracy, 0.70)
        self.assertGreater(metrics.sample_count, 500)
        self.assertLess(metrics.brier_score, metrics.baseline_brier_score)
        self.assertGreater(metrics.net_edge_bps, 0.0)
        self.assertTrue(metrics.passed_research_gate)

    def test_gate_rejects_accurate_model_that_cannot_pay_the_fees(self) -> None:
        expensive = dict(GATE, round_trip_cost_bps=500.0)
        metrics = walk_forward_backtest(predictive_dataset(), **expensive)
        # Direction accuracy is unchanged; only the cost of trading it moved.
        self.assertGreater(metrics.signal_accuracy, 0.70)
        self.assertFalse(metrics.passed_research_gate)
        self.assertTrue(any("maliyet" in reason for reason in metrics.gate_reasons))
        self.assertAlmostEqual(
            metrics.net_edge_bps, metrics.gross_edge_bps - 500.0, places=9
        )

    def test_net_edge_interval_brackets_the_mean(self) -> None:
        metrics = walk_forward_backtest(predictive_dataset(), **GATE)
        self.assertLessEqual(metrics.net_edge_ci95_low, metrics.net_edge_bps)
        self.assertGreaterEqual(metrics.net_edge_ci95_high, metrics.net_edge_bps)
        self.assertGreater(metrics.signal_days, 1)

    def test_day_block_bootstrap_is_wider_than_ignoring_clusters(self) -> None:
        rng = np.random.default_rng(7)
        # Ten days, each with its own offset: every observation inside a day
        # tells you almost the same thing, so the honest interval is wide.
        day_effect = rng.normal(0, 5.0, 10)
        days = np.repeat(np.arange(10), 50)
        values = day_effect[days] + rng.normal(0, 0.1, days.size)
        low, high = day_block_bootstrap_interval(values, days)
        naive = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
        self.assertGreater(high - low, 2 * naive)

    def test_bundle_round_trip_and_scenario_fallback(self) -> None:
        dataset = predictive_dataset()
        metrics = walk_forward_backtest(dataset, **GATE)
        bundle = fit_final_bundle(
            dataset, symbol="BTCUSDT", interval="5m", backtest=metrics
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_bundle(bundle, path)
            restored = load_bundle(path)
        self.assertAlmostEqual(
            bundle.predict_up_probability(dataset.x[-1]),
            restored.predict_up_probability(dataset.x[-1]),
            places=12,
        )
        self.assertEqual(
            restored.backtest.round_trip_cost_bps, metrics.round_trip_cost_bps
        )
        scenario = select_scenario(
            restored.scenarios, 0.99, 10.0, minimum_count=10_000
        )
        self.assertEqual(scenario["count"], restored.scenarios["global"]["count"])
        self.assertIn("touch_both_probability", scenario)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(61, 100)
        self.assertLess(low, 0.61)
        self.assertGreater(high, 0.61)


if __name__ == "__main__":
    unittest.main()
