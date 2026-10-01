from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from market_fixtures import ohlcv, with_flow

from crypto_forecaster.config import Settings
from crypto_forecaster.data import (
    BinanceMarketDataClient,
    FuturesMarketSnapshot,
    MarketDataError,
)
from crypto_forecaster.long_scout import (
    DAY,
    HOUR,
    STEP,
    LongScoutWorker,
    _metrics,
    _publish,
    _scout_lock,
    bull_history,
    candidate,
    features,
    paper_outcome,
    profile,
    root,
    run_long_scout,
    select_universe,
    settle_paper,
)


def bars(count=500, step=HOUR):
    close = np.linspace(100, 120, count)
    return with_flow(
        ohlcv(
            close=close,
            high=close * 1.005,
            low=close * 0.995,
            volume=np.full(count, 10000.0),
            start_ms=0,
            step_ms=step,
        ),
        buy_ratio=0.6,
    )


class LongScoutTests(unittest.TestCase):
    def test_zero_limit_expands_beyond_eighty_without_bypassing_quality(self):
        contracts = {f"C{i}USDT": {"onboardDate": 900 * DAY} for i in range(120)}
        snapshots = {s: FuturesMarketSnapshot(s, 100, 100.01, 1,
            funding_rate_bps=1, quote_volume_24h_usdt=20_000_000) for s in contracts}
        snapshots["C0USDT"] = replace(snapshots["C0USDT"], spread_bps=100)
        snapshots["C1USDT"] = replace(snapshots["C1USDT"], quote_volume_24h_usdt=1)
        snapshots["C2USDT"] = replace(snapshots["C2USDT"], funding_rate_bps=20)
        snapshots["C3USDT"] = replace(snapshots["C3USDT"], spread_bps=float("nan"))
        snapshots["C4USDT"] = replace(snapshots["C4USDT"], quote_volume_24h_usdt=float("inf"))
        selected = select_universe(contracts, snapshots, Settings(long_scout_universe_limit=0), 1000 * DAY)
        self.assertEqual(len(selected), 115)
        self.assertFalse({"C0USDT", "C1USDT", "C2USDT", "C3USDT", "C4USDT"}.intersection(selected))

    def test_minute_worker_ticks_do_not_repeat_five_minute_settlement(self):
        with patch("crypto_forecaster.long_scout.ThreadPoolExecutor") as executor:
            worker = LongScoutWorker(Settings(long_scout_minute_shadow_enabled=True), lambda _: None)
            executor.return_value.submit.return_value.done.return_value = True
            executor.return_value.submit.return_value.result.return_value = {}
            worker.tick(datetime.fromtimestamp(DAY / 1000, UTC))
            worker.tick(datetime.fromtimestamp((DAY + 10_000) / 1000, UTC))
            self.assertEqual(executor.return_value.submit.call_count, 1)
            worker.tick(datetime.fromtimestamp((DAY + 60_000) / 1000, UTC))
            self.assertEqual(executor.return_value.submit.call_args.kwargs, {"discover": False, "settle": False})
            worker.tick(datetime.fromtimestamp((DAY + STEP) / 1000, UTC))
            self.assertEqual(executor.return_value.submit.call_args.kwargs, {"discover": False, "settle": True})
            worker.stop()
    def test_manual_refresh_cannot_overlap_service_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), scalp_state_dir=Path(directory))
            lock = root(settings) / "worker.lock"
            client = MagicMock()
            with _scout_lock(lock) as first:
                self.assertTrue(first)
                with _scout_lock(lock) as second:
                    self.assertFalse(second)
                run_long_scout(settings, client=client)
                client.fetch_crypto_perpetual_contracts.assert_not_called()
            with _scout_lock(lock) as third:
                self.assertTrue(third)

    def test_future_prices_and_btc_changes_do_not_change_past_features(self):
        history = bars()
        btc = bars()
        prefix = features(history.iloc[:400], btc.iloc[:400])
        history.loc[400:, ["close", "high", "quote_volume"]] *= 100
        btc.loc[400:, "close"] *= 0.01
        combined = features(history, btc)
        pd.testing.assert_frame_equal(prefix, combined.iloc[:400])

    def test_exchange_gap_restarts_lookback(self):
        history = bars()
        output = features(history.drop(index=420), history)
        self.assertEqual(len(output), 79)
        self.assertFalse(output["ready"].any())

    def test_bull_requires_six_contiguous_hour_confirmations(self):
        index = np.arange(12) * HOUR + HOUR - 1
        table = pd.DataFrame(
            {"ready": True, "return24": 2.0, "macroTrend": True}, index=index
        )
        output = bull_history(
            {s: table.copy() for s in ("BTCUSDT", "ETHUSDT", "AAAUSDT")}
        )
        self.assertFalse(output.iloc[4]["establishedBull"])
        self.assertTrue(output.iloc[5]["establishedBull"])
        bearish = table.copy()
        bearish.loc[index[-1], "macroTrend"] = False
        output = bull_history({"BTCUSDT": table, "ETHUSDT": bearish, "AAAUSDT": table})
        self.assertFalse(output.iloc[-1]["establishedBull"])
        gap = table.drop(index=index[6])
        output = bull_history({s: gap for s in ("BTCUSDT", "ETHUSDT", "AAAUSDT")})
        self.assertFalse(output.iloc[-1]["establishedBull"])

    def test_liquidity_gates_apply_even_to_recent_listings(self):
        settings = Settings(long_scout_universe_limit=10)
        contracts = {
            f"C{i}USDT": {"onboardDate": DAY * (1000 - 20 - i)} for i in range(12)
        }
        snapshots = {
            s: FuturesMarketSnapshot(
                s,
                100,
                100.01,
                1,
                funding_rate_bps=1,
                quote_volume_24h_usdt=(i + 1) * 10_000_000,
            )
            for i, s in enumerate(contracts)
        }
        snapshots["C0USDT"] = replace(snapshots["C0USDT"], spread_bps=100)
        snapshots["C1USDT"] = replace(snapshots["C1USDT"], quote_volume_24h_usdt=1)
        contracts["C2USDT"]["onboardDate"] = 999 * DAY  # Not enough listing history.
        selected = select_universe(contracts, snapshots, settings, 1000 * DAY)
        self.assertNotIn("C0USDT", selected)
        self.assertNotIn("C1USDT", selected)
        self.assertNotIn("C2USDT", selected)
        self.assertEqual(selected[:2], ["C3USDT", "C4USDT"])

    def test_candidate_is_long_and_not_a_probability_or_late_pump(self):
        row = pd.Series(
            {
                "ready": True,
                "trend": True,
                "close_time_ms": 1000 * DAY,
                "close": 100.0,
                "rs7d": 6.0,
                "rs24": 2.0,
                "volume4": 2.0,
                "compression": 0.5,
                "atrPct": 1.0,
                "breakoutPct": 1.0,
                "closePosition": 0.9,
                "takerRatio": 0.6,
                "quote24": 20_000_000,
                "return24": 5.0,
                "return7d": 10.0,
                "retest": False,
            }
        )
        result = candidate(row, "AAAUSDT", {"onboardDate": 980 * DAY}, Settings())
        self.assertEqual(result["direction"], "LONG")
        self.assertEqual(result["strategy"], "LB1")
        self.assertAlmostEqual(
            result["score"], sum(result["scoreParts"].values()), places=2
        )
        self.assertNotIn("probability", result)
        row["return24"] = 40
        self.assertIsNone(
            candidate(row, "AAAUSDT", {"onboardDate": 980 * DAY}, Settings())
        )

    def test_full_future_candles_only_and_next_open_is_entry(self):
        frame = bars(count=3, step=STEP)
        frame.loc[0, "high"] = 1000  # Before decision; cannot turn into a win.
        record = {"entryOpenMs": STEP, "stopPct": 3, "costBps": 12}
        frame.loc[1, ["open", "high", "low", "close"]] = [105, 105.5, 104.5, 105]
        outcome = paper_outcome(record, frame, 2 * STEP)
        self.assertEqual(outcome["entryPrice"], 105)
        self.assertIsNone(outcome["targets"]["2"]["hit"])
        self.assertFalse(outcome["mature"])

    def test_stop_before_later_target_and_same_bar_are_distinct(self):
        frame = bars(count=3, step=STEP)
        frame.loc[:, ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.0]
        record = {"entryOpenMs": STEP, "stopPct": 3, "costBps": 12}
        frame.loc[1, "low"] = 96
        frame.loc[2, "high"] = 106
        result = paper_outcome(record, frame, 3 * STEP)
        self.assertTrue(result["targets"]["5"]["hit"])
        self.assertFalse(result["targets"]["5"]["firstBeforeStop"])
        self.assertEqual(result["targets"]["5"]["netBps"], -312)
        frame.loc[1, "high"] = 106
        result = paper_outcome(record, frame, 3 * STEP)
        self.assertIsNone(result["targets"]["5"]["firstBeforeStop"])
        self.assertTrue(result["targets"]["5"]["sameBarAmbiguous"])
        self.assertIsNone(result["targets"]["5"]["netBps"])

    def test_missing_future_is_unknown_and_early_hits_excluded_from_rates(self):
        frame = bars(count=3, step=STEP)
        record = {"entryOpenMs": STEP, "stopPct": 3, "costBps": 12}
        missing = paper_outcome(record, frame.drop(index=1), 3 * STEP)
        self.assertFalse(missing["complete"])
        early = paper_outcome(record, frame, 3 * STEP)
        summary = _metrics([{**record, "outcome": early}])
        self.assertEqual(summary["targets"]["2"]["n"], 0)
        self.assertEqual(summary["pending"], 1)

    def test_disabled_observer_never_calls_market_client(self):
        with patch("crypto_forecaster.long_scout.BinanceMarketDataClient") as client:
            self.assertEqual(run_long_scout(Settings(long_scout_enabled=False)), {})
            client.assert_not_called()

    def test_metadata_excludes_non_crypto_delivery_and_bad_symbols(self):
        item = {
            "symbol": "AAAUSDT",
            "onboardDate": 1,
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "underlyingType": "COIN",
        }
        payload = {
            "symbols": [
                item,
                {**item, "symbol": "GOLDUSDT", "underlyingType": "COMMODITY"},
                {**item, "symbol": "PAXGUSDT"},
                {**item, "symbol": "XAUTUSDT"},
                {**item, "symbol": "USDCUSDT"},
                {**item, "symbol": "AAPLUSDT", "underlyingSubType": ["STOCK"]},
                {**item, "symbol": "BBBUSD", "quoteAsset": "USD"},
                {**item, "symbol": "CCCUSDT", "contractType": "CURRENT_QUARTER"},
                {**item, "symbol": "../BADUSDT"},
            ]
        }
        client = BinanceMarketDataClient(market_name="futures")
        with patch.object(client, "_request_public_json", return_value=payload):
            self.assertEqual(
                list(client.fetch_crypto_perpetual_contracts()), ["AAAUSDT"]
            )
        with (
            patch.object(client, "_request_public_json", return_value={}),
            self.assertRaises(MarketDataError),
        ):
            client.fetch_crypto_perpetual_contracts()

    def test_expired_network_failure_is_missing_not_permanently_pending(self):
        record = {"symbol": "AAAUSDT", "entryOpenMs": STEP, "outcome": {}}
        state = {"records": [record]}
        now = datetime.fromtimestamp((8 * DAY + STEP) / 1000, UTC)
        with patch(
            "crypto_forecaster.long_scout.update_market_cache",
            side_effect=MarketDataError("offline"),
        ):
            settle_paper(Settings(), state, MagicMock(), now)
        self.assertTrue(record["outcome"]["mature"])
        self.assertFalse(record["outcome"]["complete"])
        self.assertEqual(_metrics(state["records"])["pending"], 0)
        self.assertEqual(_metrics(state["records"])["missing"], 1)

    def test_background_job_does_not_overlap_or_block_ticks(self):
        with patch("crypto_forecaster.long_scout.ThreadPoolExecutor") as executor:
            worker = LongScoutWorker(Settings(), lambda _: None)
            executor.return_value.submit.return_value.done.return_value = False
            worker.tick(datetime.fromtimestamp(DAY / 1000, UTC))
            worker.tick(datetime.fromtimestamp((DAY + STEP) / 1000, UTC))
            self.assertEqual(executor.return_value.submit.call_count, 1)
            worker.stop()
            self.assertTrue(worker.cancel.is_set())

    def test_old_setting_profiles_and_private_fields_are_not_in_public_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(scalp_state_dir=Path(directory))
            profile_id, _ = profile(settings)
            record = {
                "symbol": "AAAUSDT",
                "strategy": "LB1",
                "recordedAtMs": DAY,
                "score": 70,
                "stopPct": 3,
                "entryOpenMs": DAY + STEP,
                "profileId": profile_id,
                "privateOwnerId": 123,
                "outcome": {},
            }
            summary = _publish(settings, {"records": [record]}, DAY)
            self.assertNotIn(
                "privateOwnerId", (root(settings) / "public-summary.json").read_text()
            )
            self.assertEqual(summary["performance"]["records"], 1)
            summary = _publish(
                replace(settings, long_scout_minimum_score=80),
                {"records": [record]},
                DAY,
            )
            self.assertEqual(summary["performance"]["records"], 0)
