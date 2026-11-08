"""Direction and quality must use the same minimum sample per horizon."""
import unittest
from dataclasses import replace

from test_scalping import observation
from crypto_forecaster.scalping import scalp_forecast_stats, scalp_setup_assessment


def outcomes(symbol, count, horizons, gross):
    return [
        {"family": "F1", "perpetual_symbol": symbol, "regime_state": "BULL",
         "horizon_minutes": horizon, "gross_bps": gross, "net_bps": gross - 12,
         "round_trip_cost_bps": 12, "bar_close_time_ms": sample * 86_400_000,
         "exit_time_ms": sample * 86_400_000 + horizon * 60_000}
        for sample in range(count) for horizon in horizons
    ]


class ForecastCohortTests(unittest.TestCase):
    def setUp(self):
        self.item = replace(observation(family="F1"), regime_state="BULL")

    def test_three_horizons_do_not_turn_four_events_into_twelve_samples(self):
        rows = outcomes("BTCUSDT", 4, (15, 30, 60), -40)
        rows += outcomes("ETHUSDT", 10, (15, 30, 60), 80)
        stats = scalp_forecast_stats(self.item, rows)
        for horizon in (15, 30, 60):
            self.assertEqual(stats[horizon][0], 14)
            self.assertAlmostEqual(stats[horizon][1], 10 / 14)
        assessment = scalp_setup_assessment((self.item,), rows)
        self.assertEqual(assessment.direction, "YUKARI")
        self.assertEqual(assessment.sample_count, 14)

    def test_each_horizon_uses_its_own_available_sample(self):
        rows = outcomes("BTCUSDT", 10, (15,), -40)
        rows += outcomes("BTCUSDT", 4, (30, 60), -40)
        rows += outcomes("ETHUSDT", 10, (15, 30, 60), 80)
        stats = scalp_forecast_stats(self.item, rows)
        self.assertEqual(stats[15][0], 10)
        self.assertEqual(stats[15][1], 0)
        self.assertEqual(stats[60][0], 14)
        self.assertAlmostEqual(stats[60][1], 10 / 14)


if __name__ == "__main__":
    unittest.main()
