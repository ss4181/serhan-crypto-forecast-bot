from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from test_scalping import observation, rising_frame

from crypto_forecaster.commands import format_explanations
from crypto_forecaster.config import Settings
from crypto_forecaster.regime import (
    STEP_MS,
    deliver_regime_change,
    format_regime_change,
    format_regime_status,
    update_regime,
)
from crypto_forecaster.scalping import BullRegime, _scan_frames
from crypto_forecaster.telegram import TelegramDelivery
from crypto_forecaster.universe import load_trade1_universe

START = int(datetime(2026, 9, 17, 12, tzinfo=UTC).timestamp() * 1000) - 1
BULL = BullRegime("BULL", .95, .70, 1.0, True, 10)
WEAK = BullRegime("TRANSITION", .70, .49, 1.0, True, 10)


def at(stamp):
    return datetime.fromtimestamp(stamp / 1000, tz=UTC)


class RegimeFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            scalp_state_dir=Path(self.temp.name) / "scalp",
            telegram_state_dir=Path(self.temp.name) / "telegram",
            scalp_bull_breadth_threshold=.60,
        )
        self.path = self.settings.scalp_state_dir / "regime_status.json"

    def scan(self, index, raw=BULL, **kwargs):
        defaults = {
            "bar_close_ms": START + index * STEP_MS,
            "evaluated_at_ms": START + index * STEP_MS + 20_001,
            "healthy": True, "universe_key": "v1",
        }
        defaults.update(kwargs)
        return update_regime(self.settings, raw, **defaults)

    def state(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def enter(self):
        for index in range(3):
            self.scan(index)
        self.assertEqual(self.state()["state"], "BULL")


class RegimeTests(RegimeFixture, unittest.TestCase):
    def test_three_distinct_consecutive_candles_confirm_entry(self):
        self.assertEqual(self.scan(0).state, "TRANSITION")
        self.assertIsNone(self.state()["event"])
        self.assertEqual(self.scan(0).state, "TRANSITION")
        self.assertEqual(self.state()["count"], 1)
        self.assertEqual(self.scan(1).state, "TRANSITION")
        self.assertEqual(self.scan(2).state, "BULL")
        event = self.state()["event"]
        self.assertEqual((event["from"], event["to"]), ("TRANSITION", "BULL"))
        self.assertEqual(event["confirmations"], 3)

    def test_break_in_entry_resets_confirmations(self):
        self.scan(0)
        self.scan(1, WEAK)
        self.assertEqual(self.scan(2).state, "TRANSITION")
        self.assertEqual(self.state()["count"], 1)
        self.assertEqual(self.scan(3).state, "TRANSITION")
        self.assertEqual(self.scan(4).state, "BULL")

    def test_reconstructed_settings_reuse_persisted_confirmation(self):
        self.scan(0)
        self.settings = replace(self.settings)
        self.scan(1)
        self.assertEqual(self.scan(2).state, "BULL")

    def test_hold_band_including_fifty_percent_does_not_exit(self):
        self.enter()
        for i, breadth in enumerate((.59, .55, .50), 3):
            self.assertEqual(self.scan(i, replace(WEAK, breadth=breadth)).state, "BULL")
            self.assertEqual(self.state()["count"], 0)

    def test_ordinary_exit_needs_two_candles(self):
        self.enter()
        self.assertEqual(self.scan(3, WEAK).state, "BULL")
        self.assertEqual(self.scan(4, WEAK).state, "TRANSITION")
        self.assertEqual(self.state()["event"]["confirmations"], 2)

    def test_exit_counter_does_not_reset_when_weak_labels_alternate(self):
        self.enter()
        self.scan(3, WEAK)
        off = replace(WEAK, state="OFF", score=.40, trend_fraction=.50, persistent_up=False)
        self.assertEqual(self.scan(4, off).state, "OFF")

    def test_recovery_cancels_pending_exit(self):
        self.enter()
        self.scan(3, WEAK)
        self.scan(4, BULL)
        self.assertEqual(self.scan(5, WEAK).state, "BULL")
        self.assertEqual(self.state()["count"], 1)

    def test_major_trend_and_four_week_direction_are_required(self):
        self.enter()
        self.scan(3, replace(BULL, persistent_up=False))
        self.assertEqual(self.scan(4, replace(BULL, persistent_up=False)).state, "TRANSITION")
        self.scan(5, replace(BULL, trend_fraction=.50))
        self.scan(6, replace(BULL, trend_fraction=.50))
        self.assertNotEqual(self.scan(7, replace(BULL, trend_fraction=.50)).state, "BULL")

    def test_severe_breadth_or_one_trend_vote_exits_immediately(self):
        for changed in (replace(WEAK, breadth=.34), replace(BULL, trend_fraction=.25)):
            with self.subTest(changed=changed):
                self.path.unlink(missing_ok=True)
                self.enter()
                self.assertNotEqual(self.scan(3, changed).state, "BULL")
                self.assertEqual(self.state()["event"]["confirmations"], 1)

    def test_missing_data_suspends_without_claiming_bear_market(self):
        self.enter()
        self.assertEqual(self.scan(3, healthy=False).state, "UNKNOWN")
        text = format_regime_change(self.state()["event"])
        self.assertIn("VERİ YETERSİZ", text)
        self.assertIn("Anında koruma", text)
        self.assertIn("mevcut hedeflerin takibi sürüyor", text)
        self.assertNotEqual(self.scan(4).state, "BULL")
        self.assertNotEqual(self.scan(5).state, "BULL")
        self.assertEqual(self.scan(6).state, "BULL")

    def test_missing_data_in_same_candle_revokes_bull(self):
        self.enter()
        self.assertEqual(self.scan(2, healthy=False).state, "UNKNOWN")
        self.assertEqual(self.scan(2).state, "UNKNOWN")
        self.assertEqual(self.state()["count"], 0)

    def test_gap_requires_fresh_confirmation_and_notifies_data_gap(self):
        self.enter()
        self.assertEqual(self.scan(4).state, "UNKNOWN")
        self.assertEqual(self.state()["count"], 1)
        event = self.state()["event"]
        self.assertEqual(event["confirmations"], 0)
        self.assertIn("akışı kesildi", event["reason"])
        self.scan(5)
        self.assertEqual(self.scan(6).state, "BULL")

    def test_old_replay_never_rewinds_or_authorizes(self):
        self.enter()
        before = self.path.read_bytes()
        self.assertEqual(self.scan(1).state, "UNKNOWN")
        self.assertEqual(self.path.read_bytes(), before)

    def test_policy_or_universe_change_requires_new_entry(self):
        self.enter()
        self.assertEqual(self.scan(3, universe_key="v2").state, "TRANSITION")
        self.assertEqual(self.state()["count"], 1)
        self.assertIsNone(self.state()["event"])

    def test_entry_boundary_and_configured_hysteresis(self):
        self.settings = replace(self.settings, scalp_bull_breadth_threshold=.70)
        for i in range(3):
            self.scan(i, replace(BULL, breadth=.70))
        self.assertEqual(self.state()["state"], "BULL")
        self.assertEqual(self.scan(3, replace(WEAK, breadth=.60)).state, "BULL")
        self.scan(4, replace(WEAK, breadth=.59))
        self.assertEqual(self.scan(5, replace(WEAK, breadth=.59)).state, "TRANSITION")

    def test_nonfinite_metrics_fail_closed_and_persist_valid_json(self):
        self.enter()
        self.assertEqual(self.scan(3, replace(BULL, breadth=float("nan"))).state, "UNKNOWN")
        self.assertEqual(self.state()["breadth"], 0)

    def test_corrupt_state_is_not_trusted(self):
        self.enter()
        for raw in ("not json", "[]", '{"schema":"scalp-regime-v1"}'):
            self.path.write_text(raw, encoding="utf-8")
            self.assertIn("teyit kaydı yok", format_regime_status(self.settings))
            self.assertEqual(self.scan(4).state, "TRANSITION")
            self.assertEqual(self.state()["count"], 1)

    def test_atomic_write_failure_preserves_previous_state(self):
        self.scan(0)
        before = self.path.read_bytes()
        with patch("crypto_forecaster.regime.os.replace", side_effect=OSError("disk")), self.assertRaises(OSError):
            self.scan(1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("regime-*.tmp")), [])

    def test_status_shows_pending_and_stale_record(self):
        self.scan(0)
        text = format_regime_status(self.settings, now=at(START + 30_000))
        self.assertIn("BOĞA teyidi 1/3", text)
        self.assertIn("Trend 4/4", text)
        self.assertIn("eski", format_regime_status(self.settings, now=at(START) + timedelta(hours=1)))

    @patch("crypto_forecaster.regime.is_primary", return_value=True)
    def test_delivery_is_short_and_sent_once(self, _primary):
        self.enter()
        notifier = Mock()
        notifier.deliver_once.return_value = TelegramDelivery("SENT", 12)
        now = at(START + 2 * STEP_MS + 30_000)
        self.assertEqual(deliver_regime_change(self.settings, now=now, notifier=notifier).status, "SENT")
        self.assertIsNone(deliver_regime_change(self.settings, now=now, notifier=notifier))
        self.assertEqual(notifier.deliver_once.call_count, 1)
        text = notifier.deliver_once.call_args.kwargs["text"]
        self.assertLess(len(text), 1000)
        self.assertIn("başarı olasılığı değildir", text)

    @patch("crypto_forecaster.regime.is_primary", return_value=True)
    def test_partial_and_rejected_retry_same_id_but_uncertain_does_not(self, _primary):
        self.enter()
        notifier = Mock()
        notifier.deliver_once.side_effect = [TelegramDelivery(s, None) for s in ("PARTIAL", "REDDEDILDI", "UNCERTAIN")]
        now = at(START + 2 * STEP_MS + 30_000)
        for _ in range(4):
            deliver_regime_change(self.settings, now=now, notifier=notifier)
        self.assertEqual(notifier.deliver_once.call_count, 3)
        self.assertEqual(len({c.kwargs["signal_id"] for c in notifier.deliver_once.call_args_list}), 1)

    @patch("crypto_forecaster.regime.is_primary", return_value=True)
    def test_old_event_expires_without_sending(self, _primary):
        self.enter()
        notifier = Mock()
        self.assertIsNone(deliver_regime_change(self.settings, now=at(START) + timedelta(hours=1), notifier=notifier))
        notifier.deliver_once.assert_not_called()
        self.assertEqual(self.state()["event"]["delivery_status"], "EXPIRED")

    @patch("crypto_forecaster.regime.is_primary", return_value=False)
    def test_standby_never_sends_or_changes_event(self, _primary):
        self.enter()
        before = self.path.read_bytes()
        notifier = Mock()
        self.assertIsNone(deliver_regime_change(self.settings, notifier=notifier))
        notifier.deliver_once.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_new_transition_replaces_obsolete_pending_event(self):
        self.enter()
        old_id = self.state()["event"]["id"]
        self.scan(3, replace(WEAK, breadth=.20))
        self.assertNotEqual(self.state()["event"]["id"], old_id)
        self.assertEqual(self.state()["event"]["from"], "BULL")
        self.assertEqual(self.state()["event"]["to"], "TRANSITION")

    @patch("crypto_forecaster.regime.is_primary", return_value=True)
    def test_delivery_exception_preserves_event_for_next_pass(self, _primary):
        self.enter()
        notifier = Mock()
        notifier.deliver_once.side_effect = RuntimeError("network")
        before = self.path.read_bytes()
        with self.assertRaises(RuntimeError):
            deliver_regime_change(self.settings, now=at(START + 2 * STEP_MS + 30_000), notifier=notifier)
        self.assertEqual(self.path.read_bytes(), before)

    @patch("crypto_forecaster.regime.is_primary", return_value=True)
    def test_new_scan_during_delivery_is_not_overwritten(self, _primary):
        self.enter()
        def send(**kwargs):
            self.scan(3, healthy=False)
            return TelegramDelivery("SENT", 123)
        notifier = Mock()
        notifier.deliver_once.side_effect = send
        deliver_regime_change(self.settings, now=at(START + 2 * STEP_MS + 30_000), notifier=notifier)
        self.assertEqual(self.state()["state"], "UNKNOWN")
        self.assertEqual(self.state()["event"]["to"], "UNKNOWN")
        self.assertEqual(self.state()["event"]["delivery_status"], "PENDING")

    def test_explanations_still_fit_one_telegram_message(self):
        text = format_explanations()
        self.assertLessEqual(len(text.encode("utf-16-le")) // 2, 4000)
        self.assertIn("3 ardisik 5dk kapanis", text)

    def test_invalid_persisted_counter_cannot_shortcut_entry(self):
        self.scan(0)
        row = self.state()
        row["count"] = 999
        self.path.write_text(json.dumps(row), encoding="utf-8")
        self.assertEqual(self.scan(1).state, "TRANSITION")
        self.assertEqual(self.state()["count"], 1)


class ScannerRegimeTests(RegimeFixture, unittest.TestCase):
    # Run integration cases with the real raw-regime evaluator and state file.
    def frames(self, index=0):
        end = START + index * STEP_MS
        broad = rising_frame(600)
        delta = end - int(broad["close_time_ms"].iloc[-1])
        broad[["open_time_ms", "close_time_ms"]] += delta
        major = rising_frame(1500, step_ms=3_600_000)
        major_end = (end + 1) // 3_600_000 * 3_600_000 - 1
        delta = major_end - int(major["close_time_ms"].iloc[-1])
        major[["open_time_ms", "close_time_ms"]] += delta
        return broad, {"BTCUSDT": major, "ETHUSDT": major.copy()}

    def report(self, index=0, *, stale_count=0, major_stale=False, tracked=True):
        manifest = load_trade1_universe()
        entries = manifest.selected_entries()[:10]
        broad, majors = self.frames(index)
        if major_stale:
            majors["ETHUSDT"] = majors["ETHUSDT"].iloc[:-1]
        frames = {e.perpetual_symbol: broad.iloc[:-1] if i < stale_count else broad
                  for i, e in enumerate(entries)}
        def signals(entry, frame, **kwargs):
            return [replace(observation(family=f),
                            perpetual_symbol=entry.perpetual_symbol,
                            bar_close_time_ms=int(frame["close_time_ms"].iloc[-1]),
                            execution_eligible=True) for f in ("B1", "F3")]
        with patch("crypto_forecaster.scalping._load_major_frames", return_value=majors), \
             patch("crypto_forecaster.scalping.scan_scalp_frame", side_effect=signals), \
             patch("crypto_forecaster.scalping._relative_strength_observations", return_value=[]):
            return _scan_frames(self.settings, manifest, entries, frames,
                                errors=[], now=at(START + index * STEP_MS + 20_001),
                                snapshots={}, track_regime=tracked)

    def test_scanner_uses_confirmed_state_for_setup_gate(self):
        first = self.report(0)
        self.assertEqual(first.regime.state, "TRANSITION")
        self.assertTrue(first.observations)
        self.assertFalse(any(o.alert_tier == "KURULUM" for o in first.observations))
        self.report(1)
        third = self.report(2)
        self.assertEqual(third.regime.state, "BULL")
        self.assertTrue(all(o.alert_tier == "KURULUM" for o in third.observations))

    def test_ninety_percent_latest_candle_coverage_is_required(self):
        for i in range(3):
            self.report(i, stale_count=1)
        self.assertEqual(self.state()["state"], "BULL")
        incomplete = self.report(3, stale_count=2)
        self.assertEqual(incomplete.fresh, 10)  # generic age gate is looser
        self.assertEqual(incomplete.regime.state, "UNKNOWN")
        self.assertFalse(any(o.alert_tier == "KURULUM" for o in incomplete.observations))

    def test_stale_major_data_cannot_authorize_bull(self):
        for i in range(3):
            report = self.report(i, major_stale=True)
        self.assertEqual(report.regime.state, "UNKNOWN")

    def test_research_scan_does_not_mutate_live_confirmation(self):
        self.scan(0)
        before = self.path.read_bytes()
        self.assertEqual(self.report(1, tracked=False).regime.state, "BULL")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
