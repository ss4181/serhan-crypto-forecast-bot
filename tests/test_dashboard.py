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
    def test_directional_forward_rows_have_unique_ids_and_clear_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for symbol, setup_id, resolution, net in (
                ("BTCUSDT", "a" * 64, "DIRECTION_HIT", 12.0),
                ("ETHUSDT", "b" * 64, "DIRECTION_MISS", -4.0),
            ):
                rows.append({
                    "schema": "scalp-setup-forward-outcome-v1",
                    "setup_id": setup_id,
                    "universe_version": "test-v1",
                    "spot_symbol": symbol,
                    "perpetual_symbol": symbol,
                    "families": ["B1", "F3"],
                    "direction": "YUKARI",
                    "strategy_code": "DIRECTIONAL_LONG",
                    "strategy_label": "Trend continuation long",
                    "policy_version": "policy-v1",
                    "regime_state": "BULL",
                    "bar_close_time_ms": 1_800_000_000_000,
                    "detected_at_ms": 1_800_000_001_000,
                    "notification_sent": True,
                    "notification_sent_at_ms": 1_800_000_002_000,
                    "horizon_minutes": 15,
                    "entry_price": 100,
                    "exit_price": 100.12,
                    "exit_time_ms": 1_800_000_900_000,
                    "gross_bps": net + 8,
                    "directional_net_bps": net,
                    "round_trip_cost_bps": 8,
                    "resolution": resolution,
                    "success": resolution == "DIRECTION_HIT",
                })
            (root / "setup_forward_ledger.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            settings = Settings(scalp_state_dir=root, outcome_state_dir=root / "regular")
            payload = build_dashboard_payload(
                settings, now=datetime.fromtimestamp(1_801_000_000, tz=UTC)
            )
        forward = [row for row in payload["signals"] if row["kind"] == "scalp-forward"]
        self.assertEqual(len(forward), 2)
        self.assertNotEqual(forward[0]["signalId"], forward[1]["signalId"])
        self.assertEqual({row["status"] for row in forward}, {"YÖN DOĞRU", "YÖN YANLIŞ"})
        self.assertEqual({row["success"] for row in forward}, {True, False})
        self.assertTrue(all(row["notified"] for row in forward))

    def test_regular_data_missing_outcome_is_not_scored_as_a_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = root / "regular"
            outcomes.mkdir()
            (outcomes / "ledger.jsonl").write_text(
                json.dumps({
                    "schema": "signal-outcome-v2",
                    "signal_id": "d" * 64,
                    "symbol": "BTCUSDT",
                    "interval": "5m",
                    "direction": "YUKARI",
                    "tier": "ISLEM ADAYI",
                    "source_price": 100,
                    "source_close_time_ms": 1_700_000_000_000,
                    "resolution": "DATA_MISSING",
                    "correct": False,
                }) + "\n",
                encoding="utf-8",
            )
            settings = Settings(
                scalp_state_dir=root / "scalp", outcome_state_dir=outcomes
            )
            payload = build_dashboard_payload(
                settings, now=datetime(2026, 9, 22, tzinfo=UTC)
            )
            row = next(item for item in payload["signals"] if item["kind"] == "regular")
            self.assertEqual(row["status"], "VERİ EKSİK")
            self.assertIsNone(row["success"])

    def test_data_missing_bracket_is_unresolved_and_has_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bracket_ledger.jsonl").write_text(
                json.dumps({
                    "schema": "scalp-bracket-outcome-v1",
                    "setup_id": "f" * 64,
                    "spot_symbol": "SOLUSDT",
                    "direction": "YUKARI",
                    "source_price": 100,
                    "bar_close_time_ms": 1_700_000_000_000,
                    "horizon_minutes": 60,
                    "resolution": "DATA_MISSING",
                    "success": None,
                    "missing_reason": "CANDLE_GAP_AFTER_GRACE",
                }) + "\n",
                encoding="utf-8",
            )
            settings = Settings(scalp_state_dir=root, outcome_state_dir=root / "regular")
            payload = build_dashboard_payload(
                settings, now=datetime(2026, 9, 22, tzinfo=UTC)
            )
            row = next(item for item in payload["signals"] if item["kind"] == "scalp-bracket")
            self.assertEqual(row["status"], "VERİ EKSİK")
            self.assertIsNone(row["success"])
            self.assertEqual(row["missingReason"], "CANDLE_GAP_AFTER_GRACE")
            self.assertEqual(payload["measurements"]["audiences"]["all"][0]["unresolvedCount"], 1)

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
            quarantine = scalp / "quarantine"
            quarantine.mkdir()
            (quarantine / "target_pending__broken.json.invalid").write_text(
                '{"schema":', encoding="utf-8"
            )
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
            self.assertEqual(payload["summary"]["quarantinedRecordCount"], 1)
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
