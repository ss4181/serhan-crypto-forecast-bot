from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crypto_forecaster.cli import _deliver_cloud_eligible, _telegram_configured
from crypto_forecaster.config import Settings
from crypto_forecaster.telegram import is_primary


CREDENTIALS = {
    "CRYPTO_TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd",
    "CRYPTO_TELEGRAM_CHAT_ID": "-100123",
}


class RoleTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_primary_is_the_default(self) -> None:
        self.assertTrue(is_primary())

    @patch.dict(os.environ, {"CRYPTO_BOT_ROLE": "standby"}, clear=True)
    def test_standby_is_recognised(self) -> None:
        self.assertFalse(is_primary())

    @patch.dict(os.environ, {"CRYPTO_BOT_ROLE": "  STANDBY  "}, clear=True)
    def test_role_is_read_loosely(self) -> None:
        self.assertFalse(is_primary())

    @patch.dict(os.environ, {"CRYPTO_BOT_ROLE": "anything-else"}, clear=True)
    def test_unknown_role_still_sends(self) -> None:
        # A typo must not silence the only sender.
        self.assertTrue(is_primary())

    @patch.dict(os.environ, dict(CREDENTIALS, CRYPTO_BOT_ROLE="standby"), clear=True)
    @patch("crypto_forecaster.cli.deliver_eligible")
    def test_standby_run_never_delivers(self, deliver_mock) -> None:
        predictions = [SimpleNamespace(eligible=True)]
        self.assertEqual(_deliver_cloud_eligible(Settings(), predictions), [])
        deliver_mock.assert_not_called()
        self.assertFalse(_telegram_configured())

    @patch.dict(os.environ, dict(CREDENTIALS, CRYPTO_BOT_ROLE="primary"), clear=True)
    @patch("crypto_forecaster.cli.deliver_eligible")
    def test_primary_run_delivers(self, deliver_mock) -> None:
        deliver_mock.return_value = ["sent"]
        predictions = [SimpleNamespace(eligible=True)]
        self.assertEqual(_deliver_cloud_eligible(Settings(), predictions), ["sent"])
        deliver_mock.assert_called_once()
        self.assertTrue(_telegram_configured())


if __name__ == "__main__":
    unittest.main()
