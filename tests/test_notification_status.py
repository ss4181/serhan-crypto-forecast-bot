from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.notification_status import (
    REASONS, format_notification_status, write_notification_status,
)
from crypto_forecaster.scalping import (
    ScalpScanReport, ScalpSetupAssessment, filter_scalp_notification_report,
)
from crypto_forecaster.service import (
    _save_notification_check, answer_commands, deliver_observation_digest,
    format_runtime_status,
)
from crypto_forecaster.telegram import TelegramDelivery
from test_scalping import observation
from test_service import sample_prediction

NOW = datetime(2026, 9, 9, 12, tzinfo=UTC)
STAMP = int(NOW.timestamp() * 1000)


class NotificationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            scalp_observation_enabled=True, scalp_state_dir=Path(self.temp.name),
            scalp_minimum_alert_score=1.5,
        )
        self.items = tuple(replace(observation(family=f), alert_tier="KURULUM") for f in ("B1", "F3"))
        self.report = ScalpScanReport("v1", 89, 89, 0, (), self.items, STAMP)
        self.assessment = ScalpSetupAssessment(
            "DIRECTIONAL_LONG", "Long", "YUKARI", 60, .60, 10.0, 100, 10, .80, "ORTA",
        )

    def write(self, counts=None, status="NO_CANDIDATE", **overrides):
        values = dict(evaluated_at_ms=STAMP, fresh=89, attempted=89,
                      counts=counts or {"no_setup": 2}, delivery_status=status)
        values.update(overrides)
        write_notification_status(self.settings.scalp_state_dir, **values)

    def test_every_gate_is_counted_without_changing_the_filter(self):
        cases = [
            ("eligible", {}, self.items),
            ("no_setup", {}, self.items[:1]),
            ("direction_unclear", {"direction": "KARIŞIK"}, self.items),
            ("calibration_missing", {"success_probability": None}, self.items),
            ("probability_low", {"success_probability": .45, "expected_net_bps": -10}, self.items),
            ("expected_net_low", {"expected_net_bps": -10}, self.items),
            ("quality_low", {"quality_percentile": .40}, self.items),
            ("score_low", {"sample_count": 10}, self.items),
            ("score_low", {"quality_percentile": None}, self.items),
        ]
        for reason, overrides, items in cases:
            with self.subTest(reason=reason, overrides=overrides):
                counts = {"old_scan": 42}
                report = replace(self.report, observations=items)
                with patch("crypto_forecaster.scalping.scalp_setup_assessment", return_value=replace(self.assessment, **overrides)):
                    args = dict(minimum_score=10, minimum_quality_percentile=.60,
                                minimum_direction_probability=.55, minimum_expected_net_bps=0,
                                minimum_calibration_samples=30)
                    before = filter_scalp_notification_report(report, **args)
                    after = filter_scalp_notification_report(report, diagnostics=counts, **args)
                self.assertEqual(before, after)
                self.assertEqual(counts, {reason: 1})
                self.assertEqual(bool(after.observations), reason == "eligible")
                self.assertEqual(report.observations, items)

    def test_empty_scan_clears_previous_diagnostics(self):
        counts = {"eligible": 100}
        filter_scalp_notification_report(replace(self.report, observations=()), minimum_score=1.5, diagnostics=counts)
        self.assertEqual(counts, {})

    def test_families_count_once_and_different_symbols_separately(self):
        items = self.items + (replace(self.items[0], perpetual_symbol="ETHUSDT"),)
        counts = {}
        with patch("crypto_forecaster.scalping.scalp_setup_assessment", return_value=self.assessment):
            filter_scalp_notification_report(replace(self.report, observations=items), minimum_score=0, diagnostics=counts)
        self.assertEqual(counts, {"eligible": 1, "no_setup": 1})

    def test_quiet_status_explains_not_attempted_and_keeps_settings(self):
        self.write({"no_setup": 3, "probability_low": 1})
        text = format_notification_status(self.settings, now=NOW)
        self.assertIn("89/89", text)
        self.assertIn("0/4", text)
        self.assertIn("3 çoklu teyit yok", text)
        self.assertIn("gönderim denenmedi", text)
        self.assertIn("≥1.5", text)
        self.assertIn("≥%55", text)
        self.assertNotIn("KAYIT ESKİ", text)

    def test_old_snapshot_does_not_claim_current_service_health(self):
        self.write()
        self.assertIn("KAYIT ESKİ", format_notification_status(self.settings, now=NOW + timedelta(minutes=16)))
        self.assertNotIn("KAYIT ESKİ", format_notification_status(self.settings, now=NOW + timedelta(minutes=15)))

    def test_missing_corrupt_and_future_state_do_not_crash_commands(self):
        path = self.settings.scalp_state_dir / "notification_status.json"
        self.assertIn("henüz yok", format_notification_status(self.settings, now=NOW))
        for raw in ("not json", "[]", '{"schema":"bad"}'):
            path.write_text(raw)
            self.assertIn("okunamıyor", format_notification_status(self.settings, now=NOW))
        self.write(evaluated_at_ms=STAMP + 120_000)
        self.assertIn("okunamıyor", format_notification_status(self.settings, now=NOW))
        self.write()
        row = json.loads(path.read_text())
        row["counts"]["eligible"] = True
        path.write_text(json.dumps(row))
        self.assertIn("okunamıyor", format_notification_status(self.settings, now=NOW))

    def test_delivery_failure_partial_and_pending_are_distinct(self):
        for status, expected in (("PARTIAL", "kısmi"), ("UNCERTAIN", "belirsiz"),
                                 ("ERROR", "hata"), ("PENDING", "henüz kaydedilmedi"),
                                 ("SENT", "tamamlandı")):
            self.write({"eligible": 1}, status)
            self.assertIn(expected, format_notification_status(self.settings, now=NOW))

    def test_atomic_write_failure_preserves_previous_status_and_cleans_temp(self):
        self.write()
        previous = (self.settings.scalp_state_dir / "notification_status.json").read_bytes()
        with patch("crypto_forecaster.notification_status.os.replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.write({"eligible": 1}, "SENT")
        self.assertEqual((self.settings.scalp_state_dir / "notification_status.json").read_bytes(), previous)
        self.assertEqual(list(self.settings.scalp_state_dir.glob("*.tmp")), [])

    def test_status_write_failure_does_not_block_alert_path(self):
        messages = []
        with patch("crypto_forecaster.service.write_notification_status", side_effect=OSError("disk")):
            _save_notification_check(self.settings, self.report, {"eligible": 1}, "SENT", messages.append)
        self.assertIn("yazilamadi", messages[0])

    def test_empty_scan_and_insufficient_coverage_are_explicit(self):
        self.write(counts={key: 0 for key in REASONS}, fresh=0)
        text = format_notification_status(self.settings, now=NOW)
        self.assertIn("0/0 coin/mum", text)
        self.assertIn("kapsam yetersiz", text)

    def test_private_status_and_existing_daily_digest_show_same_diagnostics(self):
        self.write()
        predictions = [sample_prediction(symbol=symbol, interval=interval) for symbol in ("BTCUSDT", "ETHUSDT") for interval in ("5m", "15m", "1h")]
        text = format_runtime_status(self.settings, predictions, now=NOW)
        self.assertIn("SCALP BİLDİRİM DURUMU", text)
        self.assertIn("BTCUSDT", text)
        self.assertLess(len(text), 2400)
        with patch("crypto_forecaster.service.poll_and_answer") as poll:
            answer_commands(self.settings, predictions, now=NOW)
            self.assertEqual(poll.call_args.kwargs["status_text"](), text)
        with patch("crypto_forecaster.service.TelegramNotifier") as notifier:
            notifier.return_value.deliver_once.return_value = TelegramDelivery("DEDUPLICATED", None)
            deliver_observation_digest(self.settings, predictions, now=NOW)
            self.assertEqual(notifier.return_value.deliver_once.call_args.kwargs["text"], text)
        disabled = replace(self.settings, scalp_observation_enabled=False)
        self.assertNotIn("SCALP BİLDİRİM DURUMU", format_runtime_status(disabled, predictions, now=NOW))
