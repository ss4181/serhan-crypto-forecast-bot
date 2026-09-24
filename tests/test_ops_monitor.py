from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.ops_monitor import scalp_health_incidents


class OpsMonitorTests(unittest.TestCase):
    def test_public_dashboard_alarm_matches_three_hour_publish_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                scalp_state_dir=root / "state",
                report_dir=root / "reports",
                scalp_data_dir=root / "data" / "scalp",
            )
            settings.report_dir.mkdir()
            settings.scalp_data_dir.mkdir(parents=True)
            local_dashboard = settings.report_dir / "scalp-data.json"
            local_dashboard.write_text("{}", encoding="utf-8")
            cache = settings.scalp_data_dir / "BTCUSDT_5m_futures.csv"
            cache.write_text("stub", encoding="utf-8")
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            base = {
                "fresh": 89,
                "attempted": 89,
                "eligible": 1,
                "websocket_connected": True,
                "delivery_status": "SENT",
                "dashboard_url": "https://example.test/scalp-data.json",
            }

            def response_for(request, *, age_ms: int) -> BytesIO:
                if request.full_url.endswith("scalp.html"):
                    return BytesIO(b"<!doctype html><html>Trade3</html>")
                generated = datetime.fromtimestamp(
                    (now_ms - age_ms) / 1000, UTC
                ).isoformat()
                return BytesIO(json.dumps({"generatedAtUtc": generated}).encode())

            with patch(
                "crypto_forecaster.ops_monitor.urlopen",
                side_effect=lambda request, timeout: response_for(
                    request, age_ms=3 * 60 * 60_000 + 59 * 60_000
                ),
            ):
                incidents = scalp_health_incidents(
                    settings, evaluated_at_ms=now_ms, **base
                )
            self.assertNotIn("public_dashboard", {row["code"] for row in incidents})

            later_ms = now_ms + 2 * 60_000
            os.utime(local_dashboard, (later_ms / 1000,) * 2)
            os.utime(cache, (later_ms / 1000,) * 2)
            with patch(
                "crypto_forecaster.ops_monitor.urlopen",
                side_effect=lambda request, timeout: response_for(
                    request, age_ms=4 * 60 * 60_000 + 1
                ),
            ):
                incidents = scalp_health_incidents(
                    settings, evaluated_at_ms=later_ms, **base
                )
            self.assertIn("public_dashboard", {row["code"] for row in incidents})

    def test_no_candidate_is_normal_and_never_pages_owner(self) -> None:
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
            for elapsed in (0, 31 * 60_000, 12 * 60 * 60_000):
                checked_at = now + elapsed
                os.utime(settings.report_dir / "scalp-data.json", (checked_at / 1000,) * 2)
                os.utime(settings.scalp_data_dir / "BTCUSDT_5m_futures.csv", (checked_at / 1000,) * 2)
                incidents = scalp_health_incidents(
                    settings, **{**base, "evaluated_at_ms": checked_at}
                )
                self.assertNotIn("no_eligible_signal", {item["code"] for item in incidents})

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
