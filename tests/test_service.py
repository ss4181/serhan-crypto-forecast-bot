from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from crypto_forecaster.config import Settings
from crypto_forecaster.service import deliver_eligible


class ServiceTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_no_signal_does_not_require_telegram_credentials(self) -> None:
        self.assertEqual(deliver_eligible(Settings(), []), [])


if __name__ == "__main__":
    unittest.main()
