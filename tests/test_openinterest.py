from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from crypto_forecaster.openinterest import (
    OpenInterestError,
    coverage,
    load_open_interest,
    update_open_interest,
)


BASE_MS = 1_770_000_000_000


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def getcode(self) -> int:
        return 200

    def read(self, _size: int) -> bytes:
        return self.payload


def opener_for(rows: list[dict]):  # type: ignore[no-untyped-def]
    payload = json.dumps(rows).encode()

    def opener(_request, timeout):  # type: ignore[no-untyped-def]
        return FakeResponse(payload)

    return opener


def sample(count: int, *, start_ms: int = BASE_MS, first: float = 1000.0) -> list[dict]:
    return [
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": str(first + index),
            "sumOpenInterestValue": str((first + index) * 60_000),
            "timestamp": start_ms + index * 300_000,
        }
        for index in range(count)
    ]


def moment(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class OpenInterestTests(unittest.TestCase):
    def test_first_run_stores_every_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            added = update_open_interest(data, "BTCUSDT", opener=opener_for(sample(5)))
            stored = load_open_interest(data, "BTCUSDT")
        self.assertEqual(added, 5)
        self.assertEqual(len(stored), 5)
        self.assertEqual(int(stored["timestamp_ms"].iloc[0]), BASE_MS)

    def test_overlapping_page_only_appends_what_is_new(self) -> None:
        # Binance always returns the last 500 rows, so most of every page is
        # already on disk; the record must grow, not duplicate.
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            update_open_interest(data, "BTCUSDT", opener=opener_for(sample(5)))
            later = moment(BASE_MS + 5 * 300_000 + 10 * 60_000)
            added = update_open_interest(
                data, "BTCUSDT", opener=opener_for(sample(8)), now=later
            )
            stored = load_open_interest(data, "BTCUSDT")
        self.assertEqual(added, 3)
        self.assertEqual(len(stored), 8)
        self.assertEqual(list(stored["timestamp_ms"]), sorted(set(stored["timestamp_ms"])))

    def test_a_second_call_inside_the_publish_window_is_skipped(self) -> None:
        calls = 0

        def counting_opener(request, timeout):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return FakeResponse(json.dumps(sample(5)).encode())

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            soon = moment(BASE_MS + 4 * 300_000 + 60_000)
            update_open_interest(data, "BTCUSDT", opener=counting_opener, now=soon)
            added = update_open_interest(data, "BTCUSDT", opener=counting_opener, now=soon)
        self.assertEqual(added, 0)
        self.assertEqual(calls, 1)

    def test_malformed_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OpenInterestError):
                update_open_interest(
                    Path(directory), "BTCUSDT", opener=opener_for([{"timestamp": "x"}])
                )

    def test_coverage_reports_how_much_history_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            update_open_interest(data, "BTCUSDT", opener=opener_for(sample(288)))
            rows, days = coverage(data, "BTCUSDT")
        self.assertEqual(rows, 288)
        self.assertAlmostEqual(days, 287 * 300_000 / 86_400_000, places=6)

    def test_missing_file_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(load_open_interest(Path(directory), "ETHUSDT").empty)


if __name__ == "__main__":
    unittest.main()
