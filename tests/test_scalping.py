from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from crypto_forecaster.config import Settings
from crypto_forecaster.data import CSV_COLUMNS, FuturesMarketSnapshot
from crypto_forecaster.scalping import (
    BullRegime,
    ScalpObservation,
    ScalpScanReport,
    deliver_scalp_observations,
    evaluate_bull_regime,
    format_scalp_observation_digest,
    format_scalp_scorecard,
    load_scalp_ledger,
    record_scalp_observations,
    scalp_cache_path,
    scalp_scorecard,
    scan_cached_scalp_universe,
    scan_scalp_frame,
    settle_scalp_observations,
    _assign_alert_tiers,
    _relative_strength_observations,
)
from crypto_forecaster.telegram import TelegramDelivery
from crypto_forecaster.universe import UniverseEntry, load_trade1_universe


STEP_MS = 300_000
START_MS = 1_760_000_000_000


def market_frame(count: int = 320) -> pd.DataFrame:
    index = np.arange(count)
    close = 100.0 * np.exp(np.sin(index / 9.0) * 0.0008)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = 100.0 + index % 17
    opens = START_MS + index * STEP_MS
    return pd.DataFrame(
        {
            "open_time_ms": opens,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time_ms": opens + STEP_MS - 1,
            "quote_volume": volume * close,
            "trade_count": np.full(count, 50),
            "taker_buy_base": volume * 0.48,
        },
        columns=CSV_COLUMNS,
    )


def entry() -> UniverseEntry:
    return UniverseEntry("BTCUSDT", "BTCUSDT", "core30", ("S1", "S1+S4", "S2", "S3"))


def bull_regime() -> BullRegime:
    return BullRegime("BULL", 0.92, 0.72, 1.0, True, 89)


def snapshot(*, spread_bps: float = 2.0) -> FuturesMarketSnapshot:
    mid = 100.0
    half = spread_bps / 20_000.0 * mid
    return FuturesMarketSnapshot(
        "BTCUSDT", mid - half, mid + half, spread_bps, 100.0, 100.0, 1.0
    )


def rising_frame(
    count: int, *, step_ms: int = STEP_MS, gain: float = 0.0002
) -> pd.DataFrame:
    index = np.arange(count)
    close = 100.0 * np.exp(index * gain)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    volume = 100.0 + index % 11
    opens = START_MS + index * step_ms
    return pd.DataFrame(
        {
            "open_time_ms": opens,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time_ms": opens + step_ms - 1,
            "quote_volume": volume * close,
            "trade_count": np.full(count, 50),
            "taker_buy_base": volume * 0.52,
        },
        columns=CSV_COLUMNS,
    )


def observation(
    *, family: str = "F1", score: float = 1.2, close_ms: int | None = None
) -> ScalpObservation:
    frame = market_frame()
    i = 300
    return ScalpObservation(
        universe_version="2026-07-ek-g",
        spot_symbol="BTCUSDT",
        perpetual_symbol="BTCUSDT",
        universe_group="core30",
        family=family,
        score=score,
        price=float(frame.iloc[i]["close"]),
        bar_open_time_ms=int(frame.iloc[i]["open_time_ms"]),
        bar_close_time_ms=close_ms or int(frame.iloc[i]["close_time_ms"]),
        details=("log-hacim z=+3.50", "yukari kapanan bar"),
    )


class DetectorTests(unittest.TestCase):
    def test_market_wide_bull_regime_needs_persistence_trend_and_breadth(self) -> None:
        broad = {f"C{index}USDT": rising_frame(600) for index in range(12)}
        majors = {
            "BTCUSDT": rising_frame(1500, step_ms=3_600_000, gain=0.0003),
            "ETHUSDT": rising_frame(1500, step_ms=3_600_000, gain=0.00025),
        }
        regime = evaluate_bull_regime(broad, majors)
        self.assertEqual(regime.state, "BULL")
        self.assertEqual(regime.breadth, 1.0)
        self.assertTrue(regime.persistent_up)

    def test_bull_breakout_records_live_cost_without_becoming_actionable(self) -> None:
        frame = market_frame()
        previous_high = float(frame.iloc[-289:-1]["high"].max())
        frame.loc[frame.index[-1], "open"] = previous_high * 1.001
        frame.loc[frame.index[-1], "close"] = previous_high * 1.002
        frame.loc[frame.index[-1], "high"] = previous_high * 1.003
        frame.loc[frame.index[-1], "low"] = previous_high * 1.0005
        frame.loc[frame.index[-1], "volume"] = 1e9
        found = scan_scalp_frame(
            entry(),
            frame,
            universe_version="v2",
            regime=bull_regime(),
            snapshot=snapshot(spread_bps=2.0),
            settings=Settings(),
        )
        b1 = next(item for item in found if item.family == "B1")
        self.assertEqual(b1.alert_tier, "RADAR")
        self.assertTrue(b1.execution_eligible)
        self.assertAlmostEqual(b1.estimated_cost_bps, 14.0)
        self.assertEqual(b1.funding_rate_bps, 1.0)

    def test_bull_pullback_recovery_is_a_separate_research_family(self) -> None:
        frame = rising_frame(600)
        previous = frame.index[-2]
        latest = frame.index[-1]
        anchor = float(frame.iloc[-8]["close"])
        frame.loc[previous, "close"] = anchor * 0.997
        frame.loc[previous, "open"] = anchor * 1.0005
        frame.loc[previous, "high"] = anchor * 1.001
        frame.loc[previous, "low"] = anchor * 0.996
        frame.loc[latest, "open"] = anchor * 0.9975
        frame.loc[latest, "close"] = anchor * 1.002
        frame.loc[latest, "high"] = anchor * 1.0025
        frame.loc[latest, "low"] = anchor * 0.997
        frame.loc[latest, "volume"] = 1e9
        found = scan_scalp_frame(
            entry(), frame, universe_version="v2", regime=bull_regime()
        )
        self.assertIn("B2", {item.family for item in found})

    def test_relative_strength_entry_is_cross_sectional_and_edge_triggered(
        self,
    ) -> None:
        entries = tuple(
            UniverseEntry(f"C{i}USDT", f"C{i}USDT", "core30", ("S1",))
            for i in range(10)
        )
        frames = {
            item.perpetual_symbol: rising_frame(320, gain=0.00001 * (i + 1))
            for i, item in enumerate(entries)
        }
        target = entries[0].perpetual_symbol
        frame = frames[target]
        frame.loc[frame.index[-1], "close"] *= 1.05
        found = _relative_strength_observations(
            entries,
            frames,
            regime=bull_regime(),
            snapshots={},
            settings=Settings(),
            universe_version="v2",
            historical_cost_bps=12.0,
        )
        self.assertEqual(
            [(item.perpetual_symbol, item.family) for item in found], [(target, "B3")]
        )

    def test_setup_tier_needs_bull_regime_live_quote_and_two_families(self) -> None:
        frame = market_frame()
        base = observation()
        enriched = ScalpObservation(
            base.universe_version,
            base.spot_symbol,
            base.perpetual_symbol,
            base.universe_group,
            base.family,
            base.score,
            base.price,
            base.bar_open_time_ms,
            base.bar_close_time_ms,
            base.details,
            regime_state="BULL",
            spread_bps=2.0,
            execution_eligible=True,
        )
        second = ScalpObservation(
            enriched.universe_version,
            enriched.spot_symbol,
            enriched.perpetual_symbol,
            enriched.universe_group,
            "B1",
            2.0,
            float(frame.iloc[-1]["close"]),
            enriched.bar_open_time_ms,
            enriched.bar_close_time_ms,
            ("boga kirilimi",),
            regime_state="BULL",
            spread_bps=2.0,
            execution_eligible=True,
        )
        tiers = _assign_alert_tiers([enriched, second], bull_regime())
        self.assertEqual({item.alert_tier for item in tiers}, {"KURULUM"})

    def test_f1_is_an_edge_triggered_up_bar_volume_observation(self) -> None:
        frame = market_frame()
        frame.loc[frame.index[-1], "volume"] = 1e9
        frame.loc[frame.index[-1], "open"] = frame.iloc[-1]["close"] * 0.99
        found = scan_scalp_frame(entry(), frame, universe_version="v1")
        self.assertIn("F1", {item.family for item in found})
        self.assertTrue(all(item.universe_version == "v1" for item in found))

    def test_f2_detects_a_volume_confirmed_30_minute_cascade(self) -> None:
        frame = market_frame()
        frame.loc[frame.index[-1], "close"] = frame.iloc[-7]["close"] * 0.88
        frame.loc[frame.index[-1], "low"] = frame.iloc[-1]["close"] * 0.999
        frame.loc[frame.index[-1], "volume"] = 1e9
        found = scan_scalp_frame(entry(), frame, universe_version="v1")
        self.assertIn("F2", {item.family for item in found})

    def test_f3_detects_a_volume_confirmed_breakout_without_calling_it_a_trade(
        self,
    ) -> None:
        frame = market_frame()
        previous_high = float(frame.iloc[-145:-1]["high"].max())
        frame.loc[frame.index[-1], "open"] = previous_high * 1.02
        frame.loc[frame.index[-1], "close"] = previous_high * 1.01
        frame.loc[frame.index[-1], "high"] = previous_high * 1.03
        frame.loc[frame.index[-1], "low"] = previous_high * 1.005
        frame.loc[frame.index[-1], "volume"] = 1e9
        found = scan_scalp_frame(entry(), frame, universe_version="v1")
        self.assertEqual({item.family for item in found}, {"F3"})

    def test_insufficient_history_produces_no_setup(self) -> None:
        self.assertEqual(
            scan_scalp_frame(entry(), market_frame(200), universe_version="v1"), ()
        )

    def test_one_hour_cooldown_matches_the_preregistered_event_family(self) -> None:
        frame = market_frame()
        for offset in (-6, 0):
            i = frame.index[offset]
            frame.loc[i, "volume"] = 1e9
            frame.loc[i, "open"] = frame.loc[i, "close"] * 0.99
            frame.loc[i, "high"] = frame.loc[i, "close"] * 1.001
            frame.loc[i, "low"] = frame.loc[i, "open"] * 0.999
        found = scan_scalp_frame(entry(), frame, universe_version="v1")
        self.assertNotIn("F1", {item.family for item in found})

    def test_a_lagging_market_cannot_repeat_its_previous_bar_in_a_new_digest(
        self,
    ) -> None:
        manifest = load_trade1_universe()
        entries = tuple(
            item
            for item in manifest.entries
            if item.spot_symbol in {"BTCUSDT", "ETHUSDT"}
        )
        current = market_frame()
        lagging = current.iloc[:-1].copy()
        lagging.loc[lagging.index[-1], "volume"] = 1e9
        lagging.loc[lagging.index[-1], "open"] = lagging.iloc[-1]["close"] * 0.99
        lagging.loc[lagging.index[-1], "high"] = lagging.iloc[-1]["close"] * 1.001
        lagging.loc[lagging.index[-1], "low"] = lagging.iloc[-1]["open"] * 0.999
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            current.to_csv(scalp_cache_path(data_dir, "ETHUSDT"), index=False)
            lagging.to_csv(scalp_cache_path(data_dir, "BTCUSDT"), index=False)
            now_ms = int(current.iloc[-1]["close_time_ms"]) + 1_000
            report = scan_cached_scalp_universe(
                Settings(scalp_data_dir=data_dir),
                manifest=manifest,
                entries=entries,
                now=datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc),
            )
        self.assertEqual(report.fresh, 2)
        self.assertEqual(report.observations, ())


class DigestTests(unittest.TestCase):
    def test_digest_is_top_k_and_explicitly_non_actionable(self) -> None:
        manifest = load_trade1_universe()
        items = tuple(
            observation(family=family, score=score)
            for family, score in (("F1", 1.1), ("F2", 2.0), ("F3", 1.5))
        )
        report = ScalpScanReport(manifest.version, 89, 89, 0, (), items, START_MS)
        text = format_scalp_observation_digest(report, manifest=manifest, top_k=2)
        self.assertIn("F2", text)
        self.assertIn("F3", text)
        self.assertNotIn("F1 hacim", text)
        self.assertIn("ISLEM ADAYI DEGILDIR", text)
        self.assertNotIn("\n   perp ", text)
        self.assertLess(text.count("\n"), 10)
        self.assertLessEqual(len(text), 4096)

    def test_low_coverage_fails_closed_before_telegram(self) -> None:
        manifest = load_trade1_universe()
        report = ScalpScanReport(
            manifest.version, 89, 40, 0, (), (observation(),), START_MS
        )
        with self.assertRaises(RuntimeError):
            deliver_scalp_observations(Settings(), report, manifest=manifest)

    def test_delivery_uses_one_digest_not_one_message_per_symbol(self) -> None:
        class Notifier:
            def __init__(self) -> None:
                self.calls = []

            def deliver_once(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls.append(kwargs)
                return TelegramDelivery("SENT", 7)

        manifest = load_trade1_universe()
        report = ScalpScanReport(
            manifest.version, 89, 89, 0, (), (observation(),), START_MS
        )
        notifier = Notifier()
        delivery = deliver_scalp_observations(
            Settings(),
            report,
            manifest=manifest,
            notifier=notifier,  # type: ignore[arg-type]
        )
        self.assertEqual(delivery.status, "SENT")  # type: ignore[union-attr]
        self.assertEqual(len(notifier.calls), 1)


class ForwardLedgerTests(unittest.TestCase):
    def test_observation_is_settled_at_fixed_15_30_60_minute_time_exits(self) -> None:
        manifest = load_trade1_universe()
        item = observation()
        frame = market_frame()
        event_index = int(
            frame.index[frame["open_time_ms"] == item.bar_open_time_ms][0]
        )
        for offset in range(1, 13):
            frame.loc[event_index + offset, "open"] = 100.0 + offset * 0.1
            frame.loc[event_index + offset, "close"] = 100.05 + offset * 0.1
            frame.loc[event_index + offset, "high"] = 100.2 + offset * 0.1
            frame.loc[event_index + offset, "low"] = 99.9 + offset * 0.1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            data_dir = root / "data"
            data_dir.mkdir()
            frame.to_csv(scalp_cache_path(data_dir, "BTCUSDT"), index=False)
            self.assertEqual(
                record_scalp_observations(state_dir, [item], manifest=manifest), 1
            )
            now_ms = item.bar_close_time_ms + 61 * 60_000
            rows = settle_scalp_observations(
                state_dir,
                data_dir,
                now=datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc),
            )
            ledger = load_scalp_ledger(state_dir)
        self.assertEqual([row["horizon_minutes"] for row in rows], [15, 30, 60])
        self.assertEqual(len(ledger), 3)
        self.assertTrue(all(row["round_trip_cost_bps"] == 12.0 for row in rows))
        card = scalp_scorecard(
            ledger,
            now=datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc),
        )
        self.assertEqual(card["observationCount"], 1)
        self.assertIn("otomatik terfi kapisi degildir", format_scalp_scorecard(card))


if __name__ == "__main__":
    unittest.main()
