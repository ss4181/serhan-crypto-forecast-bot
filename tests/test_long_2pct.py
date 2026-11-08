from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_forecaster.config import Settings
from crypto_forecaster.data import CSV_COLUMNS
from crypto_forecaster.long_2pct import (
    format_long_2pct_status,
    record_id,
    run_long_2pct_experiment,
    select_candidates,
    settle_records,
    summarize,
    verdict,
)
from crypto_forecaster.scalping import (
    BullRegime,
    ScalpObservation,
    ScalpScanReport,
    scalp_cache_path,
)
from crypto_forecaster.telegram import TelegramDelivery

STEP = 300_000
DAY = 86_400_000
CLOSE = 1_790_000_099_999 - (1_790_000_099_999 + 1) % STEP


def observation(symbol: str, family: str, **overrides) -> ScalpObservation:
    values = {
        "universe_version": "v",
        "spot_symbol": symbol,
        "perpetual_symbol": symbol,
        "universe_group": "core",
        "family": family,
        "score": 3.0,
        "price": 100.0,
        "bar_open_time_ms": CLOSE - STEP + 1,
        "bar_close_time_ms": CLOSE,
        "details": (),
        "regime_state": "BULL",
        "execution_eligible": True,
        "alert_tier": "KURULUM",
    }
    values.update(overrides)
    return ScalpObservation(**values)


def report(*items: ScalpObservation, regime: str = "BULL", close: int = CLOSE) -> ScalpScanReport:
    return ScalpScanReport(
        "v", 89, 89, 0, (), tuple(items), close + 1, BullRegime(regime, 0, 0, 0, False, 89),
        newest_close_time_ms=close,
    )


def pair(symbol: str, score: float = 3.0, **overrides) -> tuple[ScalpObservation, ...]:
    return (
        observation(symbol, "F1", score=score, **overrides),
        observation(symbol, "B1", score=score, **overrides),
    )


def write_candles(path: Path, bars: list[tuple[float, float, float, float]], close: int = CLOSE) -> None:
    close_times = close + STEP * np.arange(1, len(bars) + 1)
    frame = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    frame["open_time_ms"] = close_times - STEP + 1
    frame["close_time_ms"] = close_times
    frame["volume"] = 1.0
    frame["quote_volume"] = 100.0
    frame["trade_count"] = 1
    frame["taker_buy_base"] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(CSV_COLUMNS)].to_csv(path, index=False)


def record(status: str, day: int = 0, net: float = 100.0) -> dict:
    return {
        "id": f"{status}{day}{net}",
        "perpetual_symbol": "X",
        "bar_close_time_ms": CLOSE + day * DAY,
        "status": status,
        "outcome": {"net_bps": net},
    }


class Notifier:
    def __init__(self, status: str = "SENT") -> None:
        self.status = status
        self.sent: list[tuple[str, str]] = []

    def deliver_owner_once(self, *, signal_id: str, text: str, state_dir: Path) -> TelegramDelivery:
        self.sent.append((signal_id, text))
        return TelegramDelivery(status=self.status, message_id=1)


class SelectionTests(unittest.TestCase):
    def test_requires_bull_two_families_setup_tier_and_eligibility(self) -> None:
        self.assertEqual(select_candidates(report(*pair("A"), regime="TRANSITION"), [], daily_limit=5), [])
        self.assertEqual(select_candidates(report(observation("A", "F1")), [], daily_limit=5), [])
        self.assertEqual(
            select_candidates(report(*pair("A", alert_tier="RADAR")), [], daily_limit=5), []
        )
        self.assertEqual(
            select_candidates(report(*pair("A", execution_eligible=False)), [], daily_limit=5), []
        )
        self.assertEqual([s for s, _ in select_candidates(report(*pair("A")), [], daily_limit=5)], ["A"])

    def test_untested_family_does_not_count_as_confirmation(self) -> None:
        items = (observation("A", "F1"), observation("A", "B4"))
        self.assertEqual(select_candidates(report(*items), [], daily_limit=5), [])

    def test_ranks_by_score_and_respects_daily_cap(self) -> None:
        items = pair("LOW", 2.0) + pair("HIGH", 9.0) + pair("MID", 4.0)
        self.assertEqual([s for s, _ in select_candidates(report(*items), [], daily_limit=2)], ["HIGH", "MID"])
        used = [
            {"id": str(i), "perpetual_symbol": f"U{i}", "bar_close_time_ms": CLOSE, "status": "STOP"}
            for i in range(4)
        ]
        self.assertEqual([s for s, _ in select_candidates(report(*items), used, daily_limit=5)], ["HIGH"])

    def test_skips_open_coin_and_already_recorded_bar(self) -> None:
        opened = [{"id": "x", "perpetual_symbol": "A", "bar_close_time_ms": CLOSE - DAY, "status": "OPEN"}]
        self.assertEqual(select_candidates(report(*pair("A")), opened, daily_limit=5), [])
        seen = [{"id": record_id("A", CLOSE), "perpetual_symbol": "A", "bar_close_time_ms": CLOSE, "status": "STOP"}]
        self.assertEqual(select_candidates(report(*pair("A")), seen, daily_limit=5), [])


class SettlementTests(unittest.TestCase):
    def open_record(self) -> dict:
        return {
            "id": "r", "perpetual_symbol": "AUSDT", "bar_close_time_ms": CLOSE, "direction": "YUKARI",
            "target_bps": 200.0, "stop_bps": 200.0, "horizon_minutes": 1440,
            "round_trip_cost_bps": 20.0, "status": "OPEN", "outcome": None,
        }

    def settle(self, bars, now_ms):
        with tempfile.TemporaryDirectory() as tmp:
            write_candles(scalp_cache_path(Path(tmp), "AUSDT"), bars)
            row = self.open_record()
            settle_records([row], Path(tmp), now_ms)
            return row

    def test_target_uses_next_open_and_subtracts_cost(self) -> None:
        row = self.settle([(100, 101, 99.5, 100.5), (100.5, 102.5, 100, 102)], CLOSE + 2 * STEP)
        self.assertEqual(row["status"], "TARGET")
        self.assertAlmostEqual(row["outcome"]["entry_price"], 100.0)
        self.assertAlmostEqual(row["outcome"]["net_bps"], np.log(1.02) * 10_000 - 20.0)

    def test_same_bar_target_and_stop_counts_as_stop(self) -> None:
        row = self.settle([(100, 103, 97, 100)], CLOSE + STEP)
        self.assertEqual(row["status"], "STOP")
        self.assertTrue(row["outcome"]["ambiguous_same_bar"])

    def test_time_exit_after_24_hours(self) -> None:
        row = self.settle([(100, 100.5, 99.5, 100.2)] * 288, CLOSE + DAY)
        self.assertEqual(row["status"], "TIME_EXIT")

    def test_stays_open_before_deadline_and_missing_after_grace(self) -> None:
        self.assertEqual(self.settle([(100, 100.5, 99.5, 100.2)], CLOSE + STEP)["status"], "OPEN")
        self.assertEqual(self.settle([], CLOSE + 4 * DAY)["status"], "DATA_MISSING")


class ScorecardTests(unittest.TestCase):
    def test_win_rate_excludes_time_exit_but_net_includes_it(self) -> None:
        rows = [record("TARGET", 0, 178.0), record("STOP", 1, -222.0), record("TIME_EXIT", 2, 10.0), record("OPEN", 3)]
        rows[-1]["outcome"] = None
        summary = summarize(rows)
        self.assertEqual((summary["wins"], summary["decided"], summary["settled"], summary["open"]), (1, 2, 3, 1))
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_net_bps"], -34.0 / 3)

    def base(self, **overrides) -> dict:
        return {"settled": 120, "signal_days": 20, "win_rate": 0.6, "mean_net_bps": 30.0,
                "mean_net_stress_bps": 20.0, "net_ci_bps": (5.0, 60.0), **overrides}

    def test_verdict_rules(self) -> None:
        start = CLOSE
        at = lambda days: start + days * DAY
        self.assertEqual(verdict(self.base(), start, at(27)), "SÜRÜYOR")
        self.assertEqual(verdict(self.base(), start, at(28)), "GEÇTİ")
        self.assertEqual(verdict(self.base(settled=99), start, at(28)), "VERİ YETERSİZ")
        self.assertEqual(verdict(self.base(signal_days=14), start, at(56)), "KAPAT")
        self.assertEqual(verdict(self.base(mean_net_bps=-1.0), start, at(28)), "BAŞARISIZ")
        self.assertEqual(verdict(self.base(win_rate=0.54), start, at(28)), "BAŞARISIZ")
        self.assertEqual(verdict(self.base(net_ci_bps=(-5.0, 60.0)), start, at(28)), "KARARSIZ")
        self.assertEqual(verdict(self.base(mean_net_stress_bps=-1.0), start, at(28)), "KARARSIZ")


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = Settings(
            scalp_state_dir=root / "state",
            scalp_data_dir=root / "data",
            telegram_state_dir=root / "telegram",
            long_2pct_experiment_enabled=True,
            scalp_alert_kill_switch=False,
        )
        self.now = datetime.fromtimestamp((CLOSE + 10_000) / 1000, UTC)

    def state(self) -> dict:
        path = self.settings.scalp_state_dir / "experiments" / "long-2pct-v1" / "state.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_disabled_writes_nothing(self) -> None:
        settings = Settings(
            scalp_state_dir=self.settings.scalp_state_dir, long_2pct_experiment_enabled=False
        )
        self.assertEqual(run_long_2pct_experiment(settings, report(*pair("A")), now=self.now, notifier=Notifier()), [])
        self.assertFalse(self.settings.scalp_state_dir.exists())

    def test_rescan_of_same_bar_sends_once(self) -> None:
        notifier = Notifier()
        for _ in range(2):
            run_long_2pct_experiment(self.settings, report(*pair("AUSDT")), now=self.now, notifier=notifier)
        self.assertEqual(len(notifier.sent), 1)
        self.assertIn("DENEYSEL LONG %2", notifier.sent[0][1])
        self.assertEqual(self.state()["records"][0]["delivery"], "SENT")

    def test_stale_report_or_kill_switch_opens_nothing(self) -> None:
        late = datetime.fromtimestamp((CLOSE + 16 * 60_000) / 1000, UTC)
        notifier = Notifier()
        run_long_2pct_experiment(self.settings, report(*pair("AUSDT")), now=late, notifier=notifier)
        self.assertEqual(notifier.sent, [])
        killed = Settings(
            scalp_state_dir=self.settings.scalp_state_dir, scalp_data_dir=self.settings.scalp_data_dir,
            telegram_state_dir=self.settings.telegram_state_dir, long_2pct_experiment_enabled=True,
            scalp_alert_kill_switch=True,
        )
        run_long_2pct_experiment(killed, report(*pair("AUSDT")), now=self.now, notifier=notifier)
        self.assertEqual(notifier.sent, [])

    def test_result_message_only_after_confirmed_first_delivery(self) -> None:
        after = datetime.fromtimestamp((CLOSE + STEP) / 1000, UTC)
        write_candles(scalp_cache_path(self.settings.scalp_data_dir, "AUSDT"), [(100, 103, 99.5, 102)])
        notifier = Notifier()
        run_long_2pct_experiment(self.settings, report(*pair("AUSDT")), now=self.now, notifier=notifier)
        run_long_2pct_experiment(self.settings, report(), now=after, notifier=notifier)
        self.assertEqual(len(notifier.sent), 2)
        self.assertIn("HEDEF", notifier.sent[1][1])

    def test_failed_first_delivery_still_measures_but_sends_no_result(self) -> None:
        after = datetime.fromtimestamp((CLOSE + STEP) / 1000, UTC)
        write_candles(scalp_cache_path(self.settings.scalp_data_dir, "AUSDT"), [(100, 103, 99.5, 102)])
        notifier = Notifier(status="UNCERTAIN")
        run_long_2pct_experiment(self.settings, report(*pair("AUSDT")), now=self.now, notifier=notifier)
        run_long_2pct_experiment(self.settings, report(), now=after, notifier=notifier)
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(self.state()["records"][0]["status"], "TARGET")

    def test_status_text_renders_empty_and_active(self) -> None:
        self.assertIn("henüz yok", format_long_2pct_status(self.settings, now=self.now))
        run_long_2pct_experiment(self.settings, report(*pair("AUSDT")), now=self.now, notifier=Notifier())
        text = format_long_2pct_status(self.settings, now=self.now)
        self.assertIn("Sinyal 1", text)
        self.assertIn("SÜRÜYOR", text)


if __name__ == "__main__":
    unittest.main()
