from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_forecaster.coin_query import (
    HORIZONS_MINUTES,
    _fit_coin_models,
    _forecast_from_cached,
    _read_model_cache,
    _refresh_query_history,
    forecast_coin,
    format_coin_forecast,
    normalize_coin_query,
)
from crypto_forecaster.config import Settings
from crypto_forecaster.data import CSV_COLUMNS, FuturesMarketSnapshot
from crypto_forecaster.features import latest_feature_vector

STEP_MS = 300_000
START_MS = 1_760_000_000_000


def bars(count: int = 2_200) -> pd.DataFrame:
    rng = np.random.default_rng(419)
    index = np.arange(count)
    returns = 0.00008 * np.sin(index / 27.0) + rng.normal(0.0, 0.0012, count)
    close = 0.1 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    wick = close * rng.uniform(0.0002, 0.002, count)
    volume = rng.lognormal(5.0, 0.35, count)
    opened = START_MS + index * STEP_MS
    return pd.DataFrame(
        {
            "open_time_ms": opened,
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": volume,
            "close_time_ms": opened + STEP_MS - 1,
            "quote_volume": volume * close,
            "trade_count": np.full(count, 25),
            "taker_buy_base": volume * 0.5,
        },
        columns=CSV_COLUMNS,
    )


class FakeMarketClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.requests = []

    def fetch_market_klines(self, symbol, interval, *, start_ms, end_ms):
        self.requests.append((symbol, interval, start_ms, end_ms))
        selected = self.frame[
            (self.frame["open_time_ms"] >= start_ms)
            & (self.frame["open_time_ms"] < end_ms)
        ]
        return selected.copy()

    def fetch_futures_market_snapshots(self):
        return {
            "ALLOUSDT": FuturesMarketSnapshot(
                "ALLOUSDT", 0.0999, 0.1001, 2.0, 0.1, 0.1, 0.2, 25_000_000.0
            )
        }


class CoinQueryTests(unittest.TestCase):
    def test_symbol_normalization_is_strict_and_path_safe(self) -> None:
        self.assertEqual(normalize_coin_query(" allo "), "ALLOUSDT")
        self.assertEqual(normalize_coin_query("ALLOUSDT"), "ALLOUSDT")
        for invalid in ("", "../ALLO", "ALLO/BTC", "ALLO USDT", "/coin ALLO"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_coin_query(invalid)

    def test_each_horizon_has_coin_specific_calibrated_oos_model(self) -> None:
        frame = bars()
        payload = _fit_coin_models(
            frame,
            symbol="ALLOUSDT",
            cost_bps=12.0,
            generated_at_ms=1_800_000_000_000,
            barrier_bps=100.0,
        )
        self.assertEqual(payload["symbol"], "ALLOUSDT")
        self.assertEqual(
            set(payload["horizons"]), {str(value) for value in HORIZONS_MINUTES}
        )
        _, vector = latest_feature_vector(frame)
        forecasts = [
            _forecast_from_cached(payload["horizons"][str(horizon)], vector, horizon)
            for horizon in HORIZONS_MINUTES
        ]
        for forecast in forecasts:
            self.assertAlmostEqual(
                forecast.probability_up + forecast.probability_down, 1.0
            )
            self.assertGreaterEqual(forecast.test_count, 1)
            self.assertGreaterEqual(forecast.accuracy_ci95[0], 0)
            self.assertLessEqual(forecast.accuracy_ci95[1], 1)
        self.assertIn(
            "15 dk",
            format_coin_forecast(
                type(
                    "F",
                    (),
                    {
                        "symbol": "ALLOUSDT",
                        "mark_price": 0.1,
                        "quote_volume_24h_usdt": 20_000_000.0,
                        "spread_bps": 2.0,
                        "funding_rate_bps": 0.2,
                        "source_close_time_ms": int(frame.iloc[-1]["close_time_ms"]),
                        "history_days": 8.0,
                        "history_rows": len(frame),
                        "round_trip_cost_bps": 12.0,
                        "forecasts": tuple(forecasts),
                    },
                )()
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            settings = Settings()
            current_ms = payload["generated_at_ms"]
            source_close_ms = payload["training_source_close_ms"]
            self.assertIsNotNone(
                _read_model_cache(
                    path,
                    "ALLOUSDT",
                    current_ms,
                    source_close_ms,
                    settings,
                    history_days=180,
                    barrier_bps=100.0,
                )
            )
            self.assertIsNone(
                _read_model_cache(
                    path,
                    "ALLOUSDT",
                    current_ms,
                    source_close_ms,
                    settings,
                    history_days=90,
                    barrier_bps=100.0,
                )
            )
            self.assertIsNone(
                _read_model_cache(
                    path,
                    "ALLOUSDT",
                    current_ms,
                    source_close_ms,
                    settings,
                    history_days=180,
                    barrier_bps=120.0,
                )
            )

    def test_existing_query_cache_backfills_history_and_updates_tail_without_gaps(
        self,
    ) -> None:
        frame = bars(2_200)
        now = datetime.fromtimestamp(
            (int(frame.iloc[-1]["close_time_ms"]) + 1) / 1000, tz=UTC
        )
        settings = Settings(coin_query_history_days=30)
        client = FakeMarketClient(frame)
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                scalp_data_dir=Path(directory), coin_query_history_days=30
            )
            cache_path = Path(directory) / "coin-query" / "ALLOUSDT_5m_futures.csv"
            cache_path.parent.mkdir(parents=True)
            frame.iloc[-100:].to_csv(cache_path, index=False)
            refreshed = _refresh_query_history(
                settings, "ALLOUSDT", client, int(now.timestamp() * 1000)
            )
        self.assertEqual(len(refreshed), len(frame))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0][0:2], ("ALLOUSDT", "5m"))
        self.assertTrue(
            np.all(np.diff(refreshed["open_time_ms"].to_numpy()) == STEP_MS)
        )

    def test_short_history_fails_instead_of_reusing_btc_eth_models(self) -> None:
        with self.assertRaises(ValueError):
            _fit_coin_models(
                bars(1_000),
                symbol="ALLOUSDT",
                cost_bps=12.0,
                generated_at_ms=1_800_000_000_000,
                barrier_bps=100.0,
            )

    def test_forecast_coin_uses_usdm_snapshot_and_persists_model_by_symbol(
        self,
    ) -> None:
        frame = bars()
        now = datetime.fromtimestamp(
            (int(frame.iloc[-1]["close_time_ms"]) + 1) / 1000, tz=UTC
        )
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                scalp_data_dir=Path(directory) / "data",
                scalp_state_dir=Path(directory) / "state",
                coin_query_history_days=30,
            )
            client = FakeMarketClient(frame)
            first = forecast_coin(settings, "ALLO", now=now, client=client)
            second = forecast_coin(settings, "ALLOUSDT", now=now, client=client)
            model_file = (
                settings.scalp_state_dir / "coin-query-models" / "ALLOUSDT.json"
            )
            model_exists = model_file.exists()
        self.assertEqual(first.symbol, "ALLOUSDT")
        self.assertEqual(first.mark_price, 0.1)
        self.assertEqual(
            [item.horizon_minutes for item in first.forecasts], list(HORIZONS_MINUTES)
        )
        for one, two in zip(first.forecasts, second.forecasts, strict=True):
            self.assertAlmostEqual(one.probability_up, two.probability_up, places=5)
            self.assertEqual(one.test_count, two.test_count)
        self.assertTrue(model_exists)
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
