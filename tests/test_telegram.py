from __future__ import annotations

from email.message import Message
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from crypto_forecaster.telegram import (
    TelegramError,
    TelegramNotifier,
    digest_signal_id,
    telegram_menu_keyboard,
)


CREDENTIALS = {
    "CRYPTO_TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd",
    "CRYPTO_TELEGRAM_CHAT_ID": "-100123",
}


def rate_limit_error(retry_after: str = "0") -> HTTPError:
    headers = Message()
    headers["Retry-After"] = retry_after
    return HTTPError("https://api.telegram.org", 429, "Too Many Requests", headers, None)


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
    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_inline_keyboard_is_sent_with_a_message(self) -> None:
        requests = []

        def opener(request, timeout):  # type: ignore[no-untyped-def]
            requests.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        keyboard = telegram_menu_keyboard()
        self.assertEqual(
            TelegramNotifier(opener=opener).send_message(
                "test", reply_markup=keyboard
            ),
            99,
        )
        self.assertEqual(requests[0]["reply_markup"], keyboard)

    def test_menu_exposes_core_and_scalp_reports(self) -> None:
        buttons = [
            button
            for row in telegram_menu_keyboard()["inline_keyboard"]
            for button in row
        ]
        callbacks = {button["callback_data"] for button in buttons}
        self.assertEqual(
            callbacks,
            {"start", "explanations", "status", "performance:30", "scalp_performance:30", "members"},
        )

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_rate_limited_send_is_retried(self) -> None:
        attempts = 0

        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise rate_limit_error()
            return FakeResponse()

        self.assertEqual(TelegramNotifier(opener=opener).send_message("test"), 99)
        self.assertEqual(attempts, 3)

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_persistent_rate_limit_gives_up_with_a_reason(self) -> None:
        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            raise rate_limit_error()

        with self.assertRaises(TelegramError) as caught:
            TelegramNotifier(opener=opener).send_message("test")
        self.assertIn("429", str(caught.exception))

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_timeout_is_never_retried(self) -> None:
        attempts = 0

        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            raise URLError("timed out")

        # The message may already have been delivered, so a second attempt
        # risks a duplicate alert.  At-most-once wins over at-least-once.
        with self.assertRaises(TelegramError):
            TelegramNotifier(opener=opener).send_message("test")
        self.assertEqual(attempts, 1)

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_a_refusal_is_retried_once_its_cause_is_fixed(self) -> None:
        # A 403 means the bot is not allowed to post: nothing was delivered, so
        # the signal must not be written off forever.  Treating it as uncertain
        # left the channel silent even after the permission was granted.
        attempts = 0

        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                headers = Message()
                raise HTTPError("https://api.telegram.org", 403, "Forbidden", headers, None)
            return FakeResponse()

        notifier = TelegramNotifier(opener=opener)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            first = notifier.deliver_once(signal_id="c" * 64, text="test", state_dir=state)
            self.assertEqual(first.status, "REDDEDILDI")
            self.assertIn("403", first.detail)
            self.assertEqual(list(state.glob("*.intent.json")), [])
            second = notifier.deliver_once(signal_id="c" * 64, text="test", state_dir=state)
        self.assertEqual(second.status, "SENT")

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_an_uncertain_failure_still_blocks_forever(self) -> None:
        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            raise URLError("timed out")

        notifier = TelegramNotifier(opener=opener)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            notifier.deliver_once(signal_id="d" * 64, text="test", state_dir=state)
            # The message may have arrived, so the intent stays and no second
            # attempt is made.
            self.assertEqual(len(list(state.glob("*.intent.json"))), 1)

    @patch.dict(os.environ, CREDENTIALS, clear=False)
    def test_failed_delivery_records_the_reason(self) -> None:
        def opener(_request, timeout):  # type: ignore[no-untyped-def]
            raise URLError("timed out")

        notifier = TelegramNotifier(opener=opener)
        with tempfile.TemporaryDirectory() as directory:
            delivery = notifier.deliver_once(
                signal_id="b" * 64, text="test", state_dir=Path(directory)
            )
        self.assertEqual(delivery.status, "UNCERTAIN")
        self.assertTrue(delivery.detail)

    def test_digest_identifier_is_stable_per_bucket(self) -> None:
        first = digest_signal_id("observation-digest", 42)
        self.assertEqual(first, digest_signal_id("observation-digest", 42))
        self.assertNotEqual(first, digest_signal_id("observation-digest", 43))
        self.assertEqual(len(first), 64)

    @patch.dict(os.environ, CREDENTIALS, clear=False)
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
