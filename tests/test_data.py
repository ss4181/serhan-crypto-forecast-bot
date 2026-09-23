from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest

import pandas as pd

from crypto_forecaster.config import cache_path
from crypto_forecaster.data import (
    BinanceKlineStream,
    ClosedKline,
    BinanceMarketDataClient,
    MarketDataError,
    append_closed_kline_cache,
    parse_closed_kline_message,
    update_cache,
    update_market_cache,
)


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
            (self.rows["open_time_ms"] >= start_ms)
            & (self.rows["close_time_ms"] < end_ms)
        ]
        return selected.reset_index(drop=True)


class StubMarketClient:
    market_name = "futures"

    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows

    def fetch_market_klines(self, symbol, interval, *, start_ms, end_ms):  # type: ignore[no-untyped-def]
        selected = self.rows[
            (self.rows["open_time_ms"] >= start_ms)
            & (self.rows["close_time_ms"] < end_ms)
        ]
        return selected.reset_index(drop=True)


class JsonResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return None

    def getcode(self) -> int:
        return 200

    def read(self, _limit: int) -> bytes:
        return self.payload


class DataTests(unittest.TestCase):
    def test_kline_stream_keeps_latest_closed_event(self) -> None:
        payload = json.dumps({
            "data": {
                "e": "kline",
                "k": {
                    "s": "BTCUSDT", "i": "5m", "x": True,
                    "t": START_MS, "T": START_MS + STEP_MS - 1,
                    "o": "100", "h": "101", "l": "99", "c": "100.5",
                    "v": "5", "q": "502.5", "n": 40, "V": "2.5",
                },
            }
        })

        class FakeSocket:
            def __init__(self) -> None:
                self.sent = False

            def recv(self) -> str | None:
                if self.sent:
                    return None
                self.sent = True
                return payload

            def close(self) -> None:
                return None

        stream = BinanceKlineStream(
            ("BTCUSDT",),
            connector=lambda _url, _timeout: FakeSocket(),
            reconnect_seconds=0.01,
        )
        stream.start()
        for _ in range(100):
            if stream.latest_closed("BTCUSDT") is not None:
                break
            time.sleep(0.001)
        stream.stop()
        self.assertEqual(stream.latest_closed("BTCUSDT").close, 100.5)  # type: ignore[union-attr]

    def test_combined_websocket_parser_accepts_only_closed_five_minute_candles(self) -> None:
        payload = {
            "stream": "btcusdt@kline_5m",
            "data": {
                "e": "kline",
                "k": {
                    "s": "BTCUSDT", "i": "5m", "x": True,
                    "t": START_MS, "T": START_MS + STEP_MS - 1,
                    "o": "100", "h": "101", "l": "99", "c": "100.5",
                    "v": "5", "q": "502.5", "n": 40, "V": "2.5",
                },
            },
        }
        result = parse_closed_kline_message(payload)
        self.assertIsInstance(result, ClosedKline)
        assert result is not None
        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertIsNone(
            parse_closed_kline_message({"data": {"e": "kline", "k": {"x": False}}})
        )

    def test_websocket_candle_appends_contiguous_cache_and_rejects_gap(self) -> None:
        opens = [START_MS + index * STEP_MS for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTCUSDT_5m_futures.csv"
            frame(opens).to_csv(path, index=False)
            now = datetime.fromtimestamp((opens[-1] + 2 * STEP_MS) / 1000, tz=timezone.utc)
            next_kline = ClosedKline(
                "BTCUSDT", opens[-1] + STEP_MS, opens[-1] + 2 * STEP_MS - 1,
                100.0, 101.0, 99.0, 100.5, 5.0, 502.5, 40, 2.5,
            )
            appended = append_closed_kline_cache(path, next_kline, days=2, now=now)
            self.assertEqual(len(appended), 13)
            gap = ClosedKline(
                "BTCUSDT", opens[-1] + 3 * STEP_MS, opens[-1] + 4 * STEP_MS - 1,
                100.0, 101.0, 99.0, 100.5, 5.0, 502.5, 40, 2.5,
            )
            with self.assertRaises(MarketDataError):
                append_closed_kline_cache(path, gap, days=2, now=now)

    def test_public_futures_snapshot_combines_spread_and_funding(self) -> None:
        payloads = iter(
            (
                [{"symbol": "BTCUSDT", "bidPrice": "99.95", "askPrice": "100.05"}],
                [
                    {
                        "symbol": "BTCUSDT",
                        "markPrice": "100.01",
                        "indexPrice": "100.00",
                        "lastFundingRate": "0.0001",
                    }
                ],
                [{"symbol": "BTCUSDT", "quoteVolume": "25000000"}],
            )
        )

        def opener(_request, *, timeout):  # type: ignore[no-untyped-def]
            self.assertEqual(timeout, 20.0)
            return JsonResponse(next(payloads))

        result = BinanceMarketDataClient(
            market_name="futures", opener=opener
        ).fetch_futures_market_snapshots()["BTCUSDT"]
        self.assertAlmostEqual(result.spread_bps, 10.0)
        self.assertAlmostEqual(result.funding_rate_bps or 0.0, 1.0)
        self.assertEqual(result.mark_price, 100.01)
        self.assertEqual(result.quote_volume_24h_usdt, 25_000_000.0)

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

    def test_broad_market_cache_is_isolated_from_the_model_cache(self) -> None:
        opens = [START_MS + index * STEP_MS for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scalp" / "TUSDT_5m_futures.csv"
            now = datetime.fromtimestamp((opens[-1] + STEP_MS) / 1000, tz=timezone.utc)
            result = update_market_cache(
                path,
                "TUSDT",
                "5m",
                days=2,
                client=StubMarketClient(frame(opens)),  # type: ignore[arg-type]
                now=now,
            )
            self.assertTrue(path.exists())
            self.assertEqual(len(result), len(opens))
            self.assertFalse((root / "TUSDT_5m_futures.csv").exists())


if __name__ == "__main__":
    unittest.main()
