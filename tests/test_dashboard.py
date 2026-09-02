from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from crypto_forecaster.config import Settings
from crypto_forecaster.dashboard import build_dashboard_payload, write_dashboard_payload


class DashboardTests(unittest.TestCase):
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
            settings = Settings(outcome_state_dir=outcomes, scalp_state_dir=scalp)
            payload = build_dashboard_payload(
                settings,
                now=datetime(2026, 9, 2, 8, 0, tzinfo=UTC),
                source_status="stale",
            )
            self.assertEqual(payload["summary"]["scalpTargetHitRate"], 1.0)
            self.assertEqual(payload["summary"]["notifiedScalpTargetHitRate"], 1.0)
            self.assertEqual(payload["sourceStatus"], "stale")
            self.assertEqual(payload["generatedAtUtc"], "2026-09-02T08:00:00Z")
            self.assertEqual(payload["latestSignalAtUtc"], "2023-11-14T22:15:00Z")
            self.assertNotIn("CRYPTO_TELEGRAM_BOT_TOKEN", json.dumps(payload))
            output = write_dashboard_payload(settings, root / "dashboard.json")
            self.assertTrue(output.exists())

    def test_dashboard_rejects_unknown_source_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "fresh veya stale"):
            build_dashboard_payload(Settings(), source_status="unknown")
