from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crypto_forecaster.cli import _deliver_cloud_eligible
from crypto_forecaster.config import Settings
from crypto_forecaster.telegram import TelegramError


class CloudCliTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    @patch("crypto_forecaster.cli.deliver_eligible")
    def test_missing_telegram_credentials_skip_delivery(self, deliver_mock) -> None:
        predictions = [SimpleNamespace(eligible=True)]
        self.assertEqual(_deliver_cloud_eligible(Settings(), predictions), [])
        deliver_mock.assert_not_called()

    @patch.dict("os.environ", {"CRYPTO_TELEGRAM_BOT_TOKEN": "configured"}, clear=True)
    def test_partial_telegram_configuration_fails_closed(self) -> None:
        with self.assertRaises(TelegramError):
            _deliver_cloud_eligible(Settings(), [SimpleNamespace(eligible=True)])


if __name__ == "__main__":
    unittest.main()
