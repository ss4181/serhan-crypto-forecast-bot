from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crypto_forecaster.config import cache_path
from crypto_forecaster.outcomes import (
    format_scorecard,
    load_ledger,
    record_delivery,
    scorecard,
    settle_pending,
)
from market_fixtures import ohlcv, with_flow


STEP_MS = 300_000
SOURCE_CLOSE_MS = 1_700_000_299_999
ENTRY = 60_000.0
BARRIER_BPS = 100.0
HORIZON_MS = 24 * 60 * 60 * 1000


def write_candles(data_dir: Path, closes: list[float], highs=None, lows=None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    close = np.asarray(closes, dtype=float)
    frame = with_flow(
        ohlcv(
            close=close,
            high=close if highs is None else np.asarray(highs, dtype=float),
            low=close if lows is None else np.asarray(lows, dtype=float),
            volume=np.full(close.size, 10.0),
            open_price=close,
            start_ms=SOURCE_CLOSE_MS + 1,
            step_ms=STEP_MS,
        )
    )
    frame.to_csv(cache_path(data_dir, "BTCUSDT", "5m"), index=False)


def park_signal(state_dir: Path, direction: str, signal_id: str = "a" * 64) -> None:
    record_delivery(
        state_dir,
        signal_id=signal_id,
        symbol="BTCUSDT",
        interval="5m",
        tier="GOZLEM",
        direction=direction,
        probability=0.62,
        source_price=ENTRY,
        source_close_time_ms=SOURCE_CLOSE_MS,
        target_close_time_ms=SOURCE_CLOSE_MS + STEP_MS,
        delivered_at_ms=SOURCE_CLOSE_MS + 500,
        barrier_bps=BARRIER_BPS,
        horizon_ms=HORIZON_MS,
    )


def settle(root: Path, *, after_ms: int):  # type: ignore[no-untyped-def]
    return settle_pending(
        root / "outcomes",
        root / "data",
        round_trip_cost_bps=10.0,
        now=datetime.fromtimestamp((SOURCE_CLOSE_MS + after_ms) / 1000, tz=timezone.utc),
    )


class BarrierSettlementTests(unittest.TestCase):
    def test_an_open_trade_is_not_scored_early(self) -> None:
        # Drifting a little is not an outcome; only a barrier or the clock is.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_candles(root / "data", [ENTRY * 1.001] * 12)
            park_signal(root / "outcomes", "YUKARI")
            self.assertEqual(settle(root, after_ms=STEP_MS * 12), [])

    def test_take_profit_pays_the_barrier_not_the_close(self) -> None:
        # The old scorecard read the next candle's close, which answers a
        # different question than the model was asked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closes = [ENTRY] * 3 + [ENTRY * 1.02] + [ENTRY] * 8
            write_candles(root / "data", closes, highs=closes)
            park_signal(root / "outcomes", "YUKARI")
            rows = settle(root, after_ms=STEP_MS * 12)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["resolution"], "HEDEF")
        self.assertTrue(rows[0]["correct"])
        self.assertAlmostEqual(rows[0]["gross_bps"], BARRIER_BPS)
        self.assertAlmostEqual(rows[0]["net_bps"], BARRIER_BPS - 10.0)

    def test_stop_costs_the_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closes = [ENTRY] * 3 + [ENTRY * 0.98] + [ENTRY] * 8
            write_candles(root / "data", closes, lows=closes)
            park_signal(root / "outcomes", "YUKARI")
            rows = settle(root, after_ms=STEP_MS * 12)
        self.assertEqual(rows[0]["resolution"], "STOP")
        self.assertFalse(rows[0]["correct"])
        self.assertAlmostEqual(rows[0]["gross_bps"], -BARRIER_BPS)

    def test_a_short_reads_the_same_candles_the_other_way(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closes = [ENTRY] * 3 + [ENTRY * 0.98] + [ENTRY] * 8
            write_candles(root / "data", closes, lows=closes)
            park_signal(root / "outcomes", "ASAGI")
            rows = settle(root, after_ms=STEP_MS * 12)
        self.assertEqual(rows[0]["resolution"], "HEDEF")
        self.assertTrue(rows[0]["correct"])
        self.assertAlmostEqual(rows[0]["gross_bps"], BARRIER_BPS)

    def test_one_candle_touching_both_is_charged_as_a_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closes = [ENTRY] * 12
            highs = list(closes)
            lows = list(closes)
            highs[3] = ENTRY * 1.02
            lows[3] = ENTRY * 0.98
            write_candles(root / "data", closes, highs=highs, lows=lows)
            park_signal(root / "outcomes", "YUKARI")
            rows = settle(root, after_ms=STEP_MS * 12)
        self.assertEqual(rows[0]["resolution"], "BELIRSIZ")
        self.assertFalse(rows[0]["correct"])
        self.assertAlmostEqual(rows[0]["gross_bps"], -BARRIER_BPS)

    def test_the_time_barrier_closes_at_market(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = HORIZON_MS // STEP_MS
            write_candles(root / "data", [ENTRY * 1.003] * count)
            park_signal(root / "outcomes", "YUKARI")
            rows = settle(root, after_ms=HORIZON_MS + STEP_MS)
        self.assertEqual(rows[0]["resolution"], "SURE")
        self.assertGreater(rows[0]["gross_bps"], 0.0)
        self.assertLess(rows[0]["gross_bps"], BARRIER_BPS)

    def test_a_settled_signal_is_not_scored_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closes = [ENTRY] * 3 + [ENTRY * 1.02] + [ENTRY] * 8
            write_candles(root / "data", closes, highs=closes)
            park_signal(root / "outcomes", "YUKARI")
            settle(root, after_ms=STEP_MS * 12)
            self.assertEqual(settle(root, after_ms=STEP_MS * 12), [])
            self.assertEqual(len(load_ledger(root / "outcomes")), 1)


class ScorecardTests(unittest.TestCase):
    def test_scorecard_reports_hit_rate_and_net_result(self) -> None:
        now = datetime.fromtimestamp((SOURCE_CLOSE_MS + 60_000) / 1000, tz=timezone.utc)
        rows = [
            {
                "schema": "signal-outcome-v2",
                "symbol": "BTCUSDT",
                "interval": "5m",
                "tier": "GOZLEM",
                "correct": correct,
                "gross_bps": gross,
                "net_bps": gross - 10.0,
                "target_close_time_ms": SOURCE_CLOSE_MS,
            }
            for correct, gross in ((True, 100.0), (True, 100.0), (False, -100.0))
        ]
        card = scorecard(rows, days=30, now=now)
        self.assertEqual(card["overall"]["count"], 3)
        self.assertAlmostEqual(card["overall"]["hitRate"], 2 / 3)
        self.assertAlmostEqual(card["overall"]["netBps"], (90.0 + 90.0 - 110.0) / 3)
        text = format_scorecard(card)
        self.assertIn("CANLI KARNE", text)
        self.assertIn("BTCUSDT", text)

    def test_empty_scorecard_is_still_a_valid_message(self) -> None:
        card = scorecard([], days=30)
        self.assertEqual(card["overall"]["count"], 0)
        self.assertIn("Henuz kapanmis sinyal yok", format_scorecard(card))


if __name__ == "__main__":
    unittest.main()
