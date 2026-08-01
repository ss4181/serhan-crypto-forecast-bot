from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crypto_forecaster.telegram import TelegramNotifier


class FakeResponse:
    def __init__(self) -> None:
        self.payload = json.dumps(
            {"ok": True, "result": {"message_id": 99, "chat": {"id": -100123}}}
        ).encode()

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def getcode(self) -> int:
        return 200

    def read(self, _size: int) -> bytes:
        return self.payload


class TelegramTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "CRYPTO_TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd",
            "CRYPTO_TELEGRAM_CHAT_ID": "-100123",
        },
        clear=False,
    )
    def test_delivery_is_deduplicated(self) -> None:
        calls = 0

        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            self.assertEqual(timeout, 20.0)
            return FakeResponse()

        notifier = TelegramNotifier(opener=opener)
        with tempfile.TemporaryDirectory() as directory:
            first = notifier.deliver_once(
                signal_id="a" * 64,
                text="test",
                state_dir=Path(directory),
            )
            second = notifier.deliver_once(
                signal_id="a" * 64,
                text="test",
                state_dir=Path(directory),
            )
        self.assertEqual(first.status, "SENT")
        self.assertEqual(second.status, "DEDUPLICATED")
        self.assertEqual(calls, 1)
        self.assertNotIn("123456", repr(notifier))


if __name__ == "__main__":
    unittest.main()
