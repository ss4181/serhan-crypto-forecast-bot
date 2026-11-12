from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.ops_monitor import (
    PUBLIC_DASHBOARD_TIMEOUT_SECONDS,
    last_dashboard_read_failures,
    mark_health_alert_sent,
    scalp_health_incidents,
)

NOW_MS = int(datetime(2026, 11, 12, 18, 0, tzinfo=UTC).timestamp() * 1000)
STEP = 300_000


def healthy(request, timeout):  # type: ignore[no-untyped-def]
    if request.full_url.split("?", 1)[0].endswith("scalp.html"):
        return BytesIO(b"<html>Trade3</html>")
    return BytesIO(json.dumps({
        "schema": "trade3-signal-dashboard-v1",
        "sourceStatus": "fresh",
        "generatedAtUtc": datetime.fromtimestamp(NOW_MS / 1000, UTC).isoformat(),
    }).encode())


def broken(request, timeout):  # type: ignore[no-untyped-def]
    raise TimeoutError("read timed out")


class DashboardReadFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = Settings(
            scalp_state_dir=root / "state",
            report_dir=root / "reports",
            scalp_data_dir=root / "data",
            scalp_pages_heartbeat_path=root / "publish.json",
        )
        self.scan = 0

    def check(self, opener) -> str | None:  # type: ignore[no-untyped-def]
        self.scan += 1
        with patch("crypto_forecaster.ops_monitor.urlopen", side_effect=opener):
            incidents = scalp_health_incidents(
                self.settings,
                evaluated_at_ms=NOW_MS + self.scan * STEP,
                fresh=89,
                attempted=89,
                eligible=0,
                websocket_connected=True,
                delivery_status=None,
                dashboard_url="https://example.test/scalp-data.json",
            )
        return next(
            (row["message"] for row in incidents if row["code"] == "public_dashboard"), None
        )

    def test_single_failures_do_not_page(self) -> None:
        self.assertIsNone(self.check(broken))
        self.assertIsNone(self.check(broken))
        self.assertEqual(last_dashboard_read_failures(self.settings), 2)

    def test_third_consecutive_failure_pages_with_the_reason(self) -> None:
        self.check(broken)
        self.check(broken)
        message = self.check(broken)
        self.assertIn("3 kontroldür okunamıyor", message or "")
        self.assertIn("TimeoutError", message or "")

    def test_success_resets_the_streak(self) -> None:
        self.check(broken)
        self.check(broken)
        self.assertIsNone(self.check(healthy))
        self.assertEqual(last_dashboard_read_failures(self.settings), 0)
        self.assertIsNone(self.check(broken))

    def test_streak_survives_a_sent_alert_and_throttle_still_applies(self) -> None:
        for _ in range(3):
            message = self.check(broken)
        self.assertIsNotNone(message)
        mark_health_alert_sent(self.settings, "public_dashboard", sent_at_ms=NOW_MS + self.scan * STEP)
        self.assertIsNone(self.check(broken))
        self.assertEqual(last_dashboard_read_failures(self.settings), 4)

    def test_requests_allow_a_slow_four_megabyte_download(self) -> None:
        timeouts: list[float] = []

        def record(request, timeout):  # type: ignore[no-untyped-def]
            timeouts.append(timeout)
            return healthy(request, timeout)

        self.check(record)
        self.assertEqual(timeouts, [PUBLIC_DASHBOARD_TIMEOUT_SECONDS] * 2)
        self.assertGreaterEqual(PUBLIC_DASHBOARD_TIMEOUT_SECONDS, 10.0)


if __name__ == "__main__":
    unittest.main()
