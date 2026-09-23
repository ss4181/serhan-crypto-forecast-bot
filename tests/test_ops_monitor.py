from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from crypto_forecaster.config import Settings
from crypto_forecaster.ops_monitor import mark_health_alert_sent, scalp_health_incidents


class OpsMonitorTests(unittest.TestCase):
    def test_no_candidate_alarm_waits_thirty_minutes_then_throttles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                scalp_state_dir=root / "state",
                report_dir=root / "reports",
                scalp_data_dir=root / "data" / "scalp",
            )
            settings.report_dir.mkdir()
            (settings.report_dir / "scalp-data.json").write_text("{}", encoding="utf-8")
            settings.scalp_data_dir.mkdir(parents=True)
            (settings.scalp_data_dir / "BTCUSDT_5m_futures.csv").write_text("stub", encoding="utf-8")
            now = int(datetime.now(UTC).timestamp() * 1000)
            base = {
                "evaluated_at_ms": now,
                "fresh": 89,
                "attempted": 89,
                "eligible": 0,
                "websocket_connected": True,
                "delivery_status": "NO_CANDIDATE",
            }
            self.assertEqual(scalp_health_incidents(settings, **base), [])
            later = {**base, "evaluated_at_ms": now + 31 * 60_000}
            os.utime(settings.report_dir / "scalp-data.json", (later["evaluated_at_ms"] / 1000,) * 2)
            os.utime(settings.scalp_data_dir / "BTCUSDT_5m_futures.csv", (later["evaluated_at_ms"] / 1000,) * 2)
            incidents = scalp_health_incidents(settings, **later)
            self.assertEqual([item["code"] for item in incidents], ["no_eligible_signal"])
            mark_health_alert_sent(settings, "no_eligible_signal", sent_at_ms=now + 31 * 60_000)
            newer = now + 60 * 60_000
            os.utime(settings.report_dir / "scalp-data.json", (newer / 1000,) * 2)
            os.utime(settings.scalp_data_dir / "BTCUSDT_5m_futures.csv", (newer / 1000,) * 2)
            self.assertEqual(scalp_health_incidents(settings, **{**later, "evaluated_at_ms": newer}), [])

    def test_binance_telegram_and_disk_incidents_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(scalp_state_dir=root / "state", report_dir=root / "reports", scalp_data_dir=root / "data" / "scalp")
            settings.scalp_state_dir.mkdir()
            settings.report_dir.mkdir()
            (settings.report_dir / "scalp-data.json").write_text("{}", encoding="utf-8")
            incidents = scalp_health_incidents(
                settings,
                evaluated_at_ms=int(datetime.now(UTC).timestamp() * 1000),
                fresh=10,
                attempted=89,
                eligible=1,
                websocket_connected=False,
                delivery_status="ERROR",
            )
            codes = {item["code"] for item in incidents}
            self.assertTrue({"binance_coverage", "websocket", "telegram"}.issubset(codes))


if __name__ == "__main__":
    unittest.main()
