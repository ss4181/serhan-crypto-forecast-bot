from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.model import BacktestMetrics
from crypto_forecaster.service import IndicatorContribution, Prediction, dashboard_snapshot, deliver_eligible


class ServiceTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_no_signal_does_not_require_telegram_credentials(self) -> None:
        self.assertEqual(deliver_eligible(Settings(), []), [])

    def test_dashboard_snapshot_explains_probability_price_and_evidence(self) -> None:
        metrics = BacktestMetrics(
            folds=4, sample_count=1000, accuracy=.54, baseline_accuracy=.50,
            brier_score=.24, baseline_brier_score=.25, expected_calibration_error=.02,
            signal_threshold=.60, signal_count=120, signal_coverage=.12,
            signal_accuracy=.64, signal_ci95_low=.55, signal_ci95_high=.71,
            signal_familywise_ci95_low=.53, signal_familywise_ci95_high=.73,
            passed_research_gate=True, gate_reasons=(),
        )
        prediction = Prediction(
            symbol="BTCUSDT", interval="15m", source_open_time_ms=1_700_000_000_000,
            source_close_time_ms=1_700_000_899_999, source_price=60_000, atr=100,
            probability_up=.64, probability_down=.36, direction="YUKARI", confidence=.64,
            target_up_price=60_050, target_down_price=59_950,
            target_up_touch_probability=.48, target_down_touch_probability=.31,
            close_range_low=59_900, close_range_median=60_020, close_range_high=60_140,
            scenario_count=250, indicators=(IndicatorContribution("RSI(14)", "42,0", "yukari yonunu destekliyor", .2),),
            backtest=metrics, eligible=True, ineligible_reasons=(),
        )
        snapshot = dashboard_snapshot([prediction])
        row = snapshot["predictions"][0]
        self.assertEqual(snapshot["schema"], "serhan-lab-snapshot-v2")
        self.assertEqual(row["probabilityUp"], .64)
        self.assertEqual(row["targetUpPrice"], 60_050)
        self.assertEqual(row["backtest"]["signalAccuracy"], .64)
        self.assertEqual(row["indicators"][0]["name"], "RSI(14)")


if __name__ == "__main__":
    unittest.main()
