from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crypto_forecaster.config import cache_path
from crypto_forecaster.data import MarketDataError, update_cache


STEP_MS = 300_000
START_MS = 1_760_000_000_000


def frame(open_times: list[int]) -> pd.DataFrame:
    count = len(open_times)
    return pd.DataFrame(
        {
            "open_time_ms": open_times,
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
            "volume": [5.0] * count,
            "close_time_ms": [item + STEP_MS - 1 for item in open_times],
            "quote_volume": [502.5] * count,
            "trade_count": [40] * count,
            "taker_buy_base": [2.5] * count,
        }
    )


class StubClient:
    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows

    def fetch_klines(self, symbol, interval, *, start_ms, end_ms):  # type: ignore[no-untyped-def]
        selected = self.rows[
            (self.rows["open_time_ms"] >= start_ms) & (self.rows["close_time_ms"] < end_ms)
        ]
        return selected.reset_index(drop=True)


class DataTests(unittest.TestCase):
    def test_halt_in_the_series_keeps_the_newest_contiguous_run(self) -> None:
        # Binance maintenance leaves a hole.  Rejecting the file outright wedged
        # the bot until someone deleted the cache by hand.
        contiguous = [START_MS + index * STEP_MS for index in range(5)]
        after_halt = [START_MS + (index + 20) * STEP_MS for index in range(6)]
        warnings: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            now = datetime.fromtimestamp(
                (after_halt[-1] + STEP_MS) / 1000, tz=timezone.utc
            )
            result = update_cache(
                data_dir,
                "BTCUSDT",
                "5m",
                days=30,
                client=StubClient(frame(contiguous + after_halt)),
                now=now,
                warn=warnings.append,
            )
            saved = pd.read_csv(cache_path(data_dir, "BTCUSDT", "5m"))
        self.assertEqual(list(result["open_time_ms"]), after_halt)
        self.assertEqual(list(saved["open_time_ms"]), after_halt)
        self.assertTrue(any("kopukluk" in item for item in warnings))

    def test_clean_series_is_kept_whole(self) -> None:
        opens = [START_MS + index * STEP_MS for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            now = datetime.fromtimestamp((opens[-1] + STEP_MS) / 1000, tz=timezone.utc)
            result = update_cache(
                Path(directory),
                "BTCUSDT",
                "5m",
                days=30,
                client=StubClient(frame(opens)),
                now=now,
            )
        self.assertEqual(list(result["open_time_ms"]), opens)

    def test_empty_response_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MarketDataError):
                update_cache(
                    Path(directory),
                    "BTCUSDT",
                    "5m",
                    days=30,
                    client=StubClient(frame([])),
                    now=datetime.now(timezone.utc),
                )


if __name__ == "__main__":
    unittest.main()
