from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crypto_forecaster.config import cache_path
from crypto_forecaster.outcomes import (
    format_scorecard,
    load_ledger,
    record_delivery,
    scorecard,
    settle_pending,
)


SOURCE_CLOSE_MS = 1_700_000_299_999
TARGET_CLOSE_MS = 1_700_000_599_999


def write_candles(data_dir: Path, closes: dict[int, float]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for close_time, close in sorted(closes.items()):
        rows.append(
            {
                "open_time_ms": close_time - 299_999,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 10.0,
                "close_time_ms": close_time,
            }
        )
    pd.DataFrame(rows).to_csv(cache_path(data_dir, "BTCUSDT", "5m"), index=False)


def park_signal(state_dir: Path, direction: str, signal_id: str = "a" * 64) -> None:
    record_delivery(
        state_dir,
        signal_id=signal_id,
        symbol="BTCUSDT",
        interval="5m",
        tier="GOZLEM",
        direction=direction,
        probability=0.62,
        source_price=60_000.0,
        source_close_time_ms=SOURCE_CLOSE_MS,
        target_close_time_ms=TARGET_CLOSE_MS,
        delivered_at_ms=SOURCE_CLOSE_MS + 500,
    )


class OutcomeTests(unittest.TestCase):
    def test_pending_signal_waits_until_its_candle_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_candles(root / "data", {TARGET_CLOSE_MS: 60_300.0})
            park_signal(root / "outcomes", "YUKARI")
            early = datetime.fromtimestamp(
                (TARGET_CLOSE_MS - 60_000) / 1000, tz=timezone.utc
            )
            self.assertEqual(
                settle_pending(
                    root / "outcomes", root / "data", round_trip_cost_bps=20.0, now=early
                ),
                [],
            )

    def test_correct_call_is_scored_net_of_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_candles(root / "data", {TARGET_CLOSE_MS: 60_300.0})
            park_signal(root / "outcomes", "YUKARI")
            later = datetime.fromtimestamp(
                (TARGET_CLOSE_MS + 60_000) / 1000, tz=timezone.utc
            )
            settled = settle_pending(
                root / "outcomes", root / "data", round_trip_cost_bps=20.0, now=later
            )
            self.assertEqual(len(settled), 1)
            row = settled[0]
            self.assertTrue(row["correct"])
            self.assertAlmostEqual(row["gross_bps"], 49.88, places=1)
            self.assertAlmostEqual(row["net_bps"], row["gross_bps"] - 20.0, places=9)
            # Settling twice must not double count.
            self.assertEqual(
                settle_pending(
                    root / "outcomes", root / "data", round_trip_cost_bps=20.0, now=later
                ),
                [],
            )
            self.assertEqual(len(load_ledger(root / "outcomes")), 1)

    def test_wrong_call_loses_more_than_the_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_candles(root / "data", {TARGET_CLOSE_MS: 60_300.0})
            park_signal(root / "outcomes", "ASAGI")
            later = datetime.fromtimestamp(
                (TARGET_CLOSE_MS + 60_000) / 1000, tz=timezone.utc
            )
            row = settle_pending(
                root / "outcomes", root / "data", round_trip_cost_bps=20.0, now=later
            )[0]
            self.assertFalse(row["correct"])
            self.assertLess(row["net_bps"], -60.0)

    def test_scorecard_reports_hit_rate_and_net_result(self) -> None:
        now = datetime.fromtimestamp((TARGET_CLOSE_MS + 60_000) / 1000, tz=timezone.utc)
        rows = [
            {
                "schema": "signal-outcome-v1",
                "symbol": "BTCUSDT",
                "interval": "5m",
                "tier": "GOZLEM",
                "correct": correct,
                "gross_bps": gross,
                "net_bps": gross - 20.0,
                "target_close_time_ms": TARGET_CLOSE_MS,
            }
            for correct, gross in ((True, 40.0), (True, 30.0), (False, -50.0))
        ]
        card = scorecard(rows, days=30, now=now)
        self.assertEqual(card["overall"]["count"], 3)
        self.assertAlmostEqual(card["overall"]["hitRate"], 2 / 3)
        self.assertAlmostEqual(card["overall"]["netBps"], (20.0 + 10.0 - 70.0) / 3)
        self.assertEqual(card["byModel"]["BTCUSDT_5m"]["count"], 3)
        text = format_scorecard(card)
        self.assertIn("CANLI KARNE", text)
        self.assertIn("BTCUSDT", text)

    def test_empty_scorecard_is_still_a_valid_message(self) -> None:
        card = scorecard([], days=30)
        self.assertEqual(card["overall"]["count"], 0)
        self.assertIn("Henuz kapanmis sinyal yok", format_scorecard(card))


if __name__ == "__main__":
    unittest.main()
