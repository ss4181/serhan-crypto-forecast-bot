from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_forecaster.config import Settings, cache_path, model_path
from crypto_forecaster.data import MarketDataError
from crypto_forecaster.features import build_supervised_dataset
from market_fixtures import ohlcv, with_flow
from crypto_forecaster.model import BacktestMetrics, fit_final_bundle, save_bundle, walk_forward_backtest
from crypto_forecaster.service import (
    IndicatorContribution,
    Prediction,
    dashboard_snapshot,
    deliver_eligible,
    evaluate_all,
    format_observation_digest,
    format_prediction,
    make_prediction,
    PredictionCache,
)


STEP_MS = 300_000


def metrics(**overrides) -> BacktestMetrics:  # type: ignore[no-untyped-def]
    base = dict(
        folds=4, sample_count=1000, accuracy=.54, baseline_accuracy=.50,
        brier_score=.24, baseline_brier_score=.25, expected_calibration_error=.02,
        signal_threshold=.60, signal_count=120, signal_coverage=.12,
        signal_accuracy=.64, signal_ci95_low=.55, signal_ci95_high=.71,
        signal_familywise_ci95_low=.53, signal_familywise_ci95_high=.73,
        round_trip_cost_bps=20.0, gross_edge_bps=26.0, net_edge_bps=6.0,
        net_edge_ci95_low=1.5, net_edge_ci95_high=10.5,
        average_win_bps=30.0, average_loss_bps=-24.0, signal_days=60,
        barrier_bps_median=40.0, barrier_horizon_candles=12,
        resolved_fraction=.82, ambiguous_fraction=.03,
        passed_research_gate=True, gate_reasons=(),
    )
    base.update(overrides)
    return BacktestMetrics(**base)  # type: ignore[arg-type]


def sample_prediction(**overrides) -> Prediction:  # type: ignore[no-untyped-def]
    base = dict(
        symbol="BTCUSDT", interval="15m", source_open_time_ms=1_700_000_000_000,
        source_close_time_ms=1_700_000_899_999, target_close_time_ms=1_700_001_799_999,
        evaluated_at_ms=1_700_000_900_500, source_price=60_000.0, atr=100.0,
        probability_up=.64, probability_down=.36, direction="YUKARI", confidence=.64,
        target_up_price=60_050.0, target_down_price=59_950.0,
        target_up_touch_probability=.48, target_down_touch_probability=.31,
        touch_both_probability=.22, touch_neither_probability=.07,
        close_range_low=59_900.0, close_range_median=60_020.0, close_range_high=60_140.0,
        scenario_count=250,
        indicators=(IndicatorContribution("RSI(14)", "42,0", "yukari yonunu destekliyor", .2),),
        backtest=metrics(), eligible=True, ineligible_reasons=(),
    )
    base.update(overrides)
    return Prediction(**base)  # type: ignore[arg-type]


def moment(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def synthetic_bars(count: int = 1600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0015, count)
    close = 60_000 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    spread = close * rng.uniform(0.0004, 0.002, count)
    return with_flow(
        ohlcv(
            close=close,
            high=np.maximum(open_price, close) + spread,
            low=np.minimum(open_price, close) - spread,
            volume=rng.lognormal(8, 0.4, count),
            open_price=open_price,
            step_ms=STEP_MS,
        ),
        seed=seed,
    )


def build_fixture(directory: Path, *, passed_gate: bool = True) -> tuple[Settings, int]:
    """Write one real CSV cache and one real model bundle for BTCUSDT 5m."""
    settings = Settings(
        data_dir=directory / "data",
        model_dir=directory / "models",
        report_dir=directory / "reports",
        telegram_state_dir=directory / "telegram",
        outcome_state_dir=directory / "outcomes",
    )
    bars = synthetic_bars()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(cache_path(settings.data_dir, "BTCUSDT", "5m"), index=False)
    dataset = build_supervised_dataset(
        bars,
        barrier_atr_multiple=settings.barrier_atr_multiple,
        barrier_horizon_candles=12,
        minimum_barrier_bps=settings.barrier_target_bps,
    )
    backtest = walk_forward_backtest(
        dataset,
        signal_threshold=settings.signal_threshold,
        minimum_signal_count=settings.minimum_signal_count,
        minimum_signal_accuracy=settings.minimum_signal_accuracy,
        maximum_ece=settings.maximum_ece,
        round_trip_cost_bps=settings.round_trip_cost_bps,
        minimum_net_edge_bps=settings.minimum_net_edge_bps,
    )
    if passed_gate:
        backtest = replace(backtest, passed_research_gate=True, gate_reasons=())
    bundle = fit_final_bundle(
        dataset, symbol="BTCUSDT", interval="5m", backtest=backtest
    )
    save_bundle(bundle, model_path(settings.model_dir, "BTCUSDT", "5m"))
    return settings, int(bars["close_time_ms"].iloc[-1])


class FreshnessTests(unittest.TestCase):
    def _reasons(self, offset_ms: int) -> tuple[str, ...]:
        with tempfile.TemporaryDirectory() as directory:
            settings, last_close_ms = build_fixture(Path(directory))
            now = datetime.fromtimestamp(
                (last_close_ms + offset_ms) / 1000, tz=timezone.utc
            )
            prediction = make_prediction(settings, "BTCUSDT", "5m", now=now)
            return prediction.ineligible_reasons

    def test_fresh_candle_has_no_timing_objection(self) -> None:
        reasons = self._reasons(1_000)
        self.assertFalse([item for item in reasons if "mum" in item and "kal" in item])
        self.assertNotIn("hedef mum zaten kapandi", reasons)

    def test_late_signal_inside_target_candle_is_blocked(self) -> None:
        reasons = self._reasons(4 * 60 * 1000)
        self.assertTrue(any("kalmali" in item for item in reasons))

    def test_signal_about_a_closed_candle_is_blocked(self) -> None:
        # The old rule allowed three whole intervals, so a delayed cloud run
        # could announce a forecast for a candle that had already closed.
        self.assertIn("hedef mum zaten kapandi", self._reasons(11 * 60 * 1000))


class MessageTests(unittest.TestCase):
    def test_trade_tier_message_leads_with_net_expectancy(self) -> None:
        text = format_prediction(sample_prediction())
        self.assertIn("ISLEM ADAYI", text)
        self.assertIn("+6.00 bps/sinyal", text)
        self.assertIn("20.0 bps gidis-donus", text)

    def test_observation_tier_states_why_it_cannot_trade(self) -> None:
        text = format_prediction(
            sample_prediction(
                eligible=False,
                ineligible_reasons=("maliyet sonrasi beklenti -14.74 bps",),
                backtest=metrics(passed_research_gate=False, net_edge_bps=-14.74),
            )
        )
        self.assertIn("GOZLEM", text)
        self.assertIn("ISLEM ADAYI DEGIL", text)

    def test_touch_probabilities_are_not_sold_as_target_and_stop(self) -> None:
        text = format_prediction(sample_prediction())
        self.assertIn("Ikisi de ayni mumda gorulur: %22.0", text)
        self.assertIn("hedef/stop cifti degildir", text)

    def test_digest_lists_every_model_even_when_none_can_trade(self) -> None:
        predictions = [
            sample_prediction(
                symbol=symbol,
                interval=interval,
                eligible=False,
                ineligible_reasons=("model walk-forward arastirma kapisini gecemedi",),
                backtest=metrics(passed_research_gate=False, net_edge_bps=-12.0),
            )
            for symbol in ("BTCUSDT", "ETHUSDT")
            for interval in ("5m", "15m", "1h")
        ]
        text = format_observation_digest(predictions, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
        self.assertIn("Islem adayi: 0 / 6", text)
        for symbol in ("BTCUSDT", "ETHUSDT"):
            for label in ("5 dakika", "15 dakika", "1 saat"):
                self.assertIn(f"{symbol} {label}", text)
        self.assertLessEqual(len(text), 4096)


class ServiceTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_no_signal_does_not_require_telegram_credentials(self) -> None:
        self.assertEqual(deliver_eligible(Settings(), []), [])

    def test_dashboard_snapshot_explains_probability_price_and_evidence(self) -> None:
        snapshot = dashboard_snapshot([sample_prediction()])
        row = snapshot["predictions"][0]
        self.assertEqual(snapshot["schema"], "serhan-lab-snapshot-v2")
        self.assertEqual(row["probabilityUp"], .64)
        self.assertEqual(row["targetUpPrice"], 60_050)
        self.assertEqual(row["tier"], "ISLEM")
        self.assertEqual(row["backtest"]["netEdgeBps"], 6.0)
        self.assertEqual(row["indicators"][0]["name"], "RSI(14)")

    def test_evaluate_all_fails_loudly_when_a_model_is_missing(self) -> None:
        # A silently skipped model would look identical to a model with nothing
        # to say, so the six-model sweep must raise instead of shrinking.
        with tempfile.TemporaryDirectory() as directory:
            settings, _ = build_fixture(Path(directory))
            with self.assertRaises(MarketDataError):
                evaluate_all(settings)


class PredictionCacheTests(unittest.TestCase):
    def test_answer_is_reused_while_its_candle_is_still_the_latest(self) -> None:
        # Recomputing indicators over a year of candles every minute saturated
        # the box, so the reuse must be real: delete the inputs and it should
        # still answer.
        with tempfile.TemporaryDirectory() as directory:
            settings, last_close_ms = build_fixture(Path(directory))
            cache = PredictionCache()
            first = make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 1_000), cache=cache
            )
            cache_path(settings.data_dir, "BTCUSDT", "5m").unlink()
            second = make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 60_000), cache=cache
            )
        self.assertEqual(first.probability_up, second.probability_up)
        self.assertEqual(first.source_close_time_ms, second.source_close_time_ms)
        self.assertEqual(second.evaluated_at_ms, last_close_ms + 60_000)

    def test_a_reused_answer_still_ages_out_of_the_timing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings, last_close_ms = build_fixture(Path(directory))
            cache = PredictionCache()
            fresh = make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 1_000), cache=cache
            )
            late = make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 4 * 60_000), cache=cache
            )
        self.assertFalse([r for r in fresh.ineligible_reasons if "kalmali" in r])
        self.assertTrue([r for r in late.ineligible_reasons if "kalmali" in r])

    def test_the_cache_expires_when_the_next_candle_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings, last_close_ms = build_fixture(Path(directory))
            cache = PredictionCache()
            make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 1_000), cache=cache
            )
            self.assertIsNone(
                cache.get("BTCUSDT", "5m", last_close_ms + STEP_MS + 1)
            )

    def test_reuse_changes_nothing_a_fresh_computation_would_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings, last_close_ms = build_fixture(Path(directory))
            when = moment(last_close_ms + 30_000)
            direct = make_prediction(settings, "BTCUSDT", "5m", now=when)
            cache = PredictionCache()
            make_prediction(
                settings, "BTCUSDT", "5m", now=moment(last_close_ms + 1_000), cache=cache
            )
            reused = make_prediction(settings, "BTCUSDT", "5m", now=when, cache=cache)
        self.assertEqual(direct, reused)

if __name__ == "__main__":
    unittest.main()
