from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from crypto_forecaster import pages_dispatch
from crypto_forecaster.config import Settings
from crypto_forecaster.ops_monitor import scalp_health_incidents

NOW_MS = int(datetime(2026, 11, 8, 20, 0, tzinfo=UTC).timestamp() * 1000)
MINUTE = 60_000
TOKEN = "github_pat_example"


class StaleDashboardAlarmTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.heartbeat = root / "publish.json"
        self.settings = Settings(
            scalp_state_dir=root / "state",
            report_dir=root / "reports",
            scalp_data_dir=root / "data",
            scalp_pages_heartbeat_path=self.heartbeat,
        )

    def public_problem(
        self, *, page_age_minutes: int, publish_age_minutes: int | None
    ) -> str | None:
        if publish_age_minutes is not None:
            self.heartbeat.write_text(
                json.dumps({"published_at_ms": NOW_MS - publish_age_minutes * MINUTE}),
                encoding="utf-8",
            )

        def respond(request, timeout):  # type: ignore[no-untyped-def]
            if request.full_url.split("?", 1)[0].endswith("scalp.html"):
                return BytesIO(b"<html>Trade3</html>")
            generated = datetime.fromtimestamp(
                (NOW_MS - page_age_minutes * MINUTE) / 1000, UTC
            ).isoformat()
            return BytesIO(json.dumps({
                "schema": "trade3-signal-dashboard-v1",
                "sourceStatus": "fresh",
                "generatedAtUtc": generated,
            }).encode())

        with patch("crypto_forecaster.ops_monitor.urlopen", side_effect=respond):
            incidents = scalp_health_incidents(
                self.settings,
                evaluated_at_ms=NOW_MS,
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

    def test_cdn_lag_with_a_confirmed_publish_does_not_page(self) -> None:
        self.assertIsNone(self.public_problem(page_age_minutes=187, publish_age_minutes=14))

    def test_very_long_cdn_lag_pages_with_the_real_cause(self) -> None:
        message = self.public_problem(page_age_minutes=7 * 60, publish_age_minutes=14)
        self.assertIn("CDN", message or "")

    def test_old_page_without_proof_of_publish_still_pages(self) -> None:
        message = self.public_problem(page_age_minutes=187, publish_age_minutes=None)
        self.assertEqual(message, "GitHub Pages dashboard verisi eski (187 dk).")

    def test_old_proof_means_publishing_really_stopped(self) -> None:
        message = self.public_problem(page_age_minutes=187, publish_age_minutes=61)
        self.assertEqual(message, "GitHub Pages dashboard verisi eski (187 dk).")

    def test_corrupt_proof_is_treated_as_missing(self) -> None:
        self.heartbeat.write_text("{bad", encoding="utf-8")
        message = self.public_problem(page_age_minutes=187, publish_age_minutes=None)
        self.assertEqual(message, "GitHub Pages dashboard verisi eski (187 dk).")

    def test_fresh_page_never_pages(self) -> None:
        self.assertIsNone(self.public_problem(page_age_minutes=20, publish_age_minutes=None))


class PublishHeartbeatTests(unittest.TestCase):
    @staticmethod
    def runs(*rows: dict) -> object:
        return lambda request, timeout: BytesIO(json.dumps({"workflow_runs": list(rows)}).encode())

    def test_reads_finish_time_of_the_newest_successful_run(self) -> None:
        opener = self.runs({"updated_at": "2026-11-08T20:01:16Z"})
        expected = int(datetime(2026, 11, 8, 20, 1, 16, tzinfo=UTC).timestamp() * 1000)
        self.assertEqual(pages_dispatch.latest_publish_ms(TOKEN, opener=opener), expected)

    def test_no_successful_run_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            pages_dispatch.latest_publish_ms(TOKEN, opener=self.runs())

    def test_rejected_lookup_never_exposes_the_token(self) -> None:
        def refuse(request, timeout):  # type: ignore[no-untyped-def]
            raise HTTPError(request.full_url, 403, "no", None, None)  # type: ignore[arg-type]

        with self.assertRaises(RuntimeError) as caught:
            pages_dispatch.latest_publish_ms(TOKEN, opener=refuse)
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_record_writes_what_the_monitor_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publish.json"
            pages_dispatch.record_publish_heartbeat(
                TOKEN, path, opener=self.runs({"updated_at": "2026-11-08T20:01:16Z"})
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(payload["published_at_ms"], int)

    def run_main(self, env: dict[str, str], **heartbeat) -> tuple[int, object, object]:
        with patch.dict("os.environ", {pages_dispatch.TOKEN_ENV: TOKEN, **env}, clear=True), patch.object(
            pages_dispatch, "dispatch_workflow"
        ) as dispatch, patch.object(
            pages_dispatch, "record_publish_heartbeat", **heartbeat
        ) as record:
            code = pages_dispatch.main()
        return code, dispatch, record

    def test_heartbeat_failure_does_not_block_the_refresh(self) -> None:
        code, dispatch, _ = self.run_main(
            {pages_dispatch.HEARTBEAT_ENV: "/x/publish.json"}, side_effect=RuntimeError("down")
        )
        self.assertEqual(code, 0)
        self.assertTrue(dispatch.called)

    def test_heartbeat_is_skipped_when_not_configured(self) -> None:
        code, _, record = self.run_main({})
        self.assertEqual(code, 0)
        self.assertFalse(record.called)

    def test_unit_grants_the_heartbeat_a_directory_and_path(self) -> None:
        unit = (
            Path(__file__).resolve().parents[1] / "deploy/trade3-pages-dispatch.service"
        ).read_text(encoding="utf-8")
        self.assertIn("StateDirectory=trade3-pages", unit)
        self.assertIn(
            "TRADE3_PAGES_HEARTBEAT_PATH=/var/lib/trade3-pages/publish.json", unit
        )
        self.assertEqual(
            str(Settings().scalp_pages_heartbeat_path.as_posix()),
            "/var/lib/trade3-pages/publish.json",
        )


if __name__ == "__main__":
    unittest.main()
