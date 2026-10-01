from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from crypto_forecaster.config import Settings, validate_interval
from crypto_forecaster.data import update_market_cache
from crypto_forecaster.growth_watchlist import (
    format_growth_watchlist,
    load_growth_watchlist,
)
from crypto_forecaster.long_scout import DAY, HOUR, STEP, root
from crypto_forecaster.minute_shadow import (
    MINUTE,
    load_minute_shadow_summary,
    outcome,
    run_minute_shadow,
    trigger,
)


def bars(count=120, start=DAY):
    opens = start + np.arange(count) * MINUTE
    close = np.linspace(100, 100.03, count)
    return pd.DataFrame(
        {
            "open_time_ms": opens,
            "close_time_ms": opens + MINUTE - 1,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "quote_volume": 1000.0,
            "volume": 10.0,
            "trade_count": 5,
            "taker_buy_base": 6.0,
        }
    )


class MinuteShadowTests(unittest.TestCase):
    def test_trigger_uses_closed_fresh_contiguous_data_and_does_not_chase(self):
        frame = bars()
        now = DAY + len(frame) * MINUTE
        frame.loc[117:, "quote_volume"] = 2000
        frame.loc[119, ["high", "low", "close"]] = [100.31, 100.05, 100.3]
        self.assertIsNotNone(trigger(frame, now))
        future = bars(1, now)
        future.loc[0, ["high", "low", "close"]] = [110, 109, 110]
        self.assertEqual(trigger(frame, now), trigger(pd.concat([frame, future]), now))
        self.assertIsNone(trigger(frame.drop(index=90), now))
        self.assertIsNone(trigger(frame, now + 2 * MINUTE))
        frame.loc[119, ["high", "low", "close"]] = [110.01, 109.8, 110]
        self.assertIsNone(trigger(frame, now))

    def test_actual_decision_time_and_next_open_are_causal(self):
        frame = bars(70)
        frame.loc[:1, "high"] = 1000  # All happened before either simulated entry.
        frame.loc[2, ["open", "high", "low", "close"]] = [100, 100.1, 99.9, 100]
        frame.loc[5, ["open", "high", "low", "close"]] = [101, 101.1, 100.9, 101]
        record = {"recordedAtMs": DAY + MINUTE + 10_000, "stopPct": 3, "costBps": 12}
        one = outcome(record, frame, DAY + 70 * MINUTE, MINUTE)
        five = outcome(record, frame, DAY + 70 * MINUTE, STEP)
        self.assertEqual(one["entryOpenMs"], DAY + 2 * MINUTE)
        self.assertEqual(five["entryOpenMs"], DAY + STEP)
        self.assertEqual(one["entryPrice"], 100)
        self.assertEqual(five["entryPrice"], 101)
        self.assertFalse(one["targets"]["2"]["hit"])
        self.assertTrue(one["mature"])
        self.assertTrue(five["complete"])

    def test_missing_and_bad_length_candles_are_not_counted(self):
        frame = bars(70)
        record = {"recordedAtMs": DAY, "stopPct": 3, "costBps": 12}
        now = DAY + 70 * MINUTE
        self.assertFalse(outcome(record, frame.drop(index=10), now, MINUTE)["complete"])
        frame.loc[10, "close_time_ms"] += 1
        self.assertFalse(outcome(record, frame, now, MINUTE)["complete"])

    def test_stop_first_and_same_candle_are_not_reported_as_profitable(self):
        frame = bars(70)
        record = {"recordedAtMs": DAY, "stopPct": 3, "costBps": 12}
        frame.loc[1, "low"] = 96
        frame.loc[2, "high"] = 106
        two = outcome(record, frame, DAY + 70 * MINUTE, MINUTE)["targets"]["2"]
        self.assertTrue(two["hit"])
        self.assertFalse(two["firstBeforeStop"])
        self.assertEqual(two["netBps"], -312)
        frame.loc[1, "high"] = 106
        two = outcome(record, frame, DAY + 70 * MINUTE, MINUTE)["targets"]["2"]
        self.assertIsNone(two["firstBeforeStop"])
        self.assertIsNone(two["netBps"])

    def test_disabled_and_waiting_regime_do_not_fetch_or_notify(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), scalp_state_dir=Path(directory))
            client = MagicMock()
            self.assertEqual(
                run_minute_shadow(
                    replace(settings, long_scout_minute_shadow_enabled=False),
                    {},
                    client=client,
                ),
                {},
            )
            result = run_minute_shadow(
                settings, {"regime": {"state": "WAIT_BULL"}}, client=client
            )
            self.assertEqual(result["candidateCount"], 0)
            self.assertEqual(result["pairedN"], 0)
            self.assertFalse(result["alerts"])
            client.fetch_market_klines.assert_not_called()

    def test_prospective_pairing_cooldown_profile_isolation_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), scalp_state_dir=Path(directory))
            frame = bars(90)
            stamp = DAY + 10 * MINUTE + 10_000
            summary = {
                "scannedAtMs": stamp,
                "regime": {"state": "BULL_CONFIRMED", "dataHealthy": True},
                "watchlist": [
                    {
                        "symbol": "AAAUSDT",
                        "score": 70,
                        "strategy": "LB1",
                        "atrPct": 1,
                        "spreadBps": 1,
                    }
                ],
            }
            clock = lambda x: datetime.fromtimestamp(x / 1000, UTC)
            with (
                patch(
                    "crypto_forecaster.minute_shadow.update_market_cache",
                    return_value=frame,
                ),
                patch(
                    "crypto_forecaster.minute_shadow.trigger",
                    return_value={"sourceCloseMs": stamp - 10_001, "price": 100},
                ),
            ):
                initial = run_minute_shadow(
                    settings, summary, now=clock(stamp), client=MagicMock()
                )
                second = run_minute_shadow(
                    settings, summary, now=clock(stamp + MINUTE), client=MagicMock()
                )
            self.assertEqual(initial["records"], 1)
            self.assertEqual(second["records"], 1)  # No per-minute duplication.
            self.assertEqual(
                second["pairedN"], 0
            )  # Early touches cannot inflate denominator.
            now = stamp + 70 * MINUTE
            with patch(
                "crypto_forecaster.minute_shadow.update_market_cache",
                return_value=frame,
            ):
                mature = run_minute_shadow(
                    settings, {}, now=clock(now), client=MagicMock()
                )
                other = run_minute_shadow(
                    replace(settings, long_scout_minimum_score=80),
                    {},
                    now=clock(now),
                    client=MagicMock(),
                )
            self.assertEqual(mature["pairedN"], 1)
            self.assertEqual(mature["entryMetrics"]["oneMinute"]["2"]["n"], 1)
            self.assertIsNotNone(mature["entryMetrics"]["oneMinute"]["2"]["wilson95"])
            self.assertEqual(other["records"], 0)
            with patch(
                "crypto_forecaster.minute_shadow.update_market_cache",
                side_effect=RuntimeError("offline"),
            ):
                # Cached paired result survives later outage. It is not falsely reclassified.
                result = run_minute_shadow(
                    settings, {}, now=clock(now), client=MagicMock()
                )
            self.assertEqual(result["pairedN"], 1)
            payload = load_minute_shadow_summary(settings)
            self.assertNotIn("tracked", payload)
            state_path = root(settings) / "minute-shadow" / "state.json"
            state = json.loads(state_path.read_text())
            state["records"][0]["oneMinute"] = {}
            state_path.write_text(json.dumps(state))
            with patch(
                "crypto_forecaster.minute_shadow.update_market_cache",
                side_effect=RuntimeError("offline"),
            ):
                result = run_minute_shadow(
                    settings, {}, now=clock(stamp + 3 * HOUR), client=MagicMock()
                )
            self.assertEqual(result["missing"], 1)
            self.assertEqual(result["pairedN"], 0)
            self.assertEqual(result["pending"], 0)

    def test_minute_caches_are_supported_without_adding_model_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = bars(100)
            client = MagicMock()
            client.fetch_market_klines.return_value = frame
            client.market_name = "futures"
            result = update_market_cache(
                Path(directory) / "AAAUSDT_1m.csv",
                "AAAUSDT",
                "1m",
                days=1,
                client=client,
                now=datetime.fromtimestamp((DAY + 100 * MINUTE) / 1000, UTC),
            )
            self.assertEqual(len(result), 100)
            self.assertEqual(client.fetch_market_klines.call_args.args[1], "1m")
        with self.assertRaises(ValueError):
            validate_interval("1m")

    def test_curated_watchlist_has_ten_sources_and_no_probability_claim(self):
        data = load_growth_watchlist()
        coins = data["coins"]
        self.assertEqual(len(coins), 10)
        self.assertEqual(len({c["symbol"] for c in coins}), 10)
        for row in coins:
            self.assertTrue(row["symbol"].endswith("USDT"))
            self.assertGreaterEqual(row["maxSupplyM"], row["circulatingM"])
            self.assertTrue(
                row["marketSource"].startswith("https://www.coingecko.com/en/coins/")
            )
            self.assertTrue(row["projectSource"].startswith("https://"))
        self.assertLess(len(format_growth_watchlist()), 4096)
        self.assertIn("sabit veri", format_growth_watchlist())
        self.assertIn("100x tahmini veya alım sinyali DEĞİL", format_growth_watchlist())
