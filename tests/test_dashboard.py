from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.dashboard import (
    build_dashboard_payload,
    dashboard_payload_text,
    write_dashboard_payload,
)


class DashboardTests(unittest.TestCase):
    def test_replayed_ledger_rows_are_counted_once_and_cap_disables_rates(self) -> None:
        row = {
            "schema": "scalp-target-outcome-v1", "setup_id": "e" * 64,
            "spot_symbol": "SOLUSDT", "direction": "YUKARI", "source_price": 100,
            "bar_close_time_ms": 1_700_000_100_000, "horizon_ms": 86_400_000,
            "target_percent": 2, "hit": True, "notification_sent": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target_ledger.jsonl").write_text(
                "\n".join(json.dumps(row) for _ in range(2)) + "\n", encoding="utf-8"
            )
            settings = Settings(scalp_state_dir=root, outcome_state_dir=root / "regular")
            now = datetime(2026, 9, 7, tzinfo=UTC)
            full = build_dashboard_payload(settings, limit=1, now=now)
            c = full["measurements"]["audiences"]["notified"][0]
            self.assertEqual(c["recordCount"], 1)
            self.assertEqual(c["hitRate"], 1)
            with patch("crypto_forecaster.dashboard.HISTORY_LIMIT", 2):
                capped = build_dashboard_payload(settings, limit=1, now=now)
            self.assertTrue(capped["historyLimitReached"])
            self.assertIsNone(capped["measurements"]["audiences"]["notified"][0]["hitRate"])

    def test_dashboard_is_redacted_and_includes_target_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = root / "outcomes"
            scalp = root / "scalp"
            outcomes.mkdir()
            scalp.mkdir()
            (outcomes / "ledger.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "signal-outcome-v2",
                        "signal_id": "a" * 64,
                        "symbol": "BTCUSDT",
                        "interval": "5m",
                        "direction": "YUKARI",
                        "tier": "ISLEM ADAYI",
                        "source_price": 100,
                        "source_close_time_ms": 1_700_000_000_000,
                        "resolution": "HEDEF",
                        "correct": True,
                        "net_bps": 80,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (scalp / "target_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "scalp-target-outcome-v1",
                        "setup_id": "b" * 64,
                        "spot_symbol": "SOLUSDT",
                        "direction": "YUKARI",
                        "source_price": 100,
                        "bar_close_time_ms": 1_700_000_100_000,
                        "target_percent": 2.0,
                        "hit": True,
                        "notification_sent": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (scalp / "bracket_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "schema": "scalp-bracket-outcome-v1",
                        "setup_id": "d" * 64,
                        "spot_symbol": "SOLUSDT",
                        "direction": "YUKARI",
                        "strategy_label": "Boğa devamı LONG",
                        "source_price": 100,
                        "bar_close_time_ms": 1_700_000_300_000,
                        "target_bps": 60.0,
                        "stop_bps": 40.0,
                        "resolution": "TARGET",
                        "net_bps": 48.0,
                        "notification_sent": True,
                        "quality_percentile": 0.8,
                        "confidence": "ORTA",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pending = scalp / "target_pending"
            pending.mkdir()
            (pending / f"{'c' * 64}.json").write_text(
                json.dumps(
                    {
                        "schema": "scalp-target-pending-v1",
                        "setup_id": "c" * 64,
                        "spot_symbol": "ENAUSDT",
                        "perpetual_symbol": "ENAUSDT",
                        "direction": "AŞAĞI",
                        "families": ["B1", "F3"],
                        "score": 2.8,
                        "source_price": 0.15,
                        "bar_close_time_ms": 1_700_000_200_000,
                        "horizon_ms": 3_600_000,
                        "probability_up": {"60": 0.35},
                        "probability_down": {"60": 0.65},
                        "notification_sent": True,
                        "delivered_percents": [],
                        "outcome_recorded_percents": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = Settings(outcome_state_dir=outcomes, scalp_state_dir=scalp)
            payload = build_dashboard_payload(
                settings,
                now=datetime(2026, 9, 2, 8, 0, tzinfo=UTC),
                source_status="stale",
            )
            self.assertIsNone(payload["summary"]["scalpTargetHitRate"])
            self.assertIsNone(payload["summary"]["notifiedScalpTargetHitRate"])
            self.assertEqual(payload["summary"]["scalpTargetCount"], 4)
            self.assertEqual(payload["summary"]["settledScalpTargetCount"], 1)
            self.assertEqual(payload["summary"]["pendingScalpTargetCount"], 3)
            self.assertEqual(payload["summary"]["targetLevels"]["2"]["hits"], 1)
            self.assertEqual(payload["summary"]["targetLevels"]["5"]["pending"], 1)
            self.assertIsNone(payload["summary"]["scalpBracketWinRate"])
            self.assertEqual(payload["summary"]["settledScalpBracketCount"], 1)
            self.assertEqual(payload["sourceStatus"], "stale")
            self.assertEqual(payload["generatedAtUtc"], "2026-09-02T08:00:00Z")
            self.assertEqual(payload["latestSignalAtUtc"], "2023-11-14T22:18:20Z")
            self.assertNotIn("CRYPTO_TELEGRAM_BOT_TOKEN", json.dumps(payload))
            rendered = json.loads(dashboard_payload_text(settings))["signals"]
            self.assertTrue(any(row["status"] == "VERİ EKSİK" for row in rendered))
            limited = build_dashboard_payload(settings, limit=1)
            self.assertEqual(limited["displayedCount"], 1)
            self.assertEqual(limited["summary"], build_dashboard_payload(settings)["summary"])
            self.assertEqual(len(limited["signals"]), 1)
            output = write_dashboard_payload(settings, root / "dashboard.json")
            self.assertTrue(output.exists())

    def test_dashboard_rejects_unknown_source_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "fresh veya stale"):
            build_dashboard_payload(Settings(), source_status="unknown")
