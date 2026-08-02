from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from crypto_forecaster.config import display_zone, local_text


# 2026-08-02 14:19:41 UTC, the moment a report that looked three hours stale
# was actually sent.
INSTANT_MS = 1_785_680_381_000


class DisplayTimeTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_default_clock_is_the_reader_s_own(self) -> None:
        # Showing 14:19 to someone whose phone reads 17:19 makes a current
        # report look old, which is how this surfaced.
        self.assertIn("17:19", local_text(INSTANT_MS))
        self.assertIn("UTC+03:00", local_text(INSTANT_MS))

    @patch.dict(os.environ, {"CRYPTO_DISPLAY_TIMEZONE": "UTC"}, clear=True)
    def test_the_clock_is_configurable(self) -> None:
        self.assertIn("14:19", local_text(INSTANT_MS))

    @patch.dict(os.environ, {"CRYPTO_DISPLAY_TIMEZONE": "Mars/Olympus"}, clear=True)
    def test_an_unknown_zone_falls_back_instead_of_crashing(self) -> None:
        self.assertIsNotNone(display_zone())
        self.assertIn("14:19", local_text(INSTANT_MS))

    @patch.dict(os.environ, {}, clear=True)
    def test_minutes_only_form_drops_seconds(self) -> None:
        self.assertNotIn(":41", local_text(INSTANT_MS, with_seconds=False))
        self.assertIn(":41", local_text(INSTANT_MS))

    @patch.dict(os.environ, {}, clear=True)
    def test_every_stamp_says_which_clock_it_used(self) -> None:
        # An unlabelled time is the whole problem; never print a bare one.
        self.assertRegex(local_text(INSTANT_MS), r"\(UTC[+-]\d{2}:\d{2}\)$")


if __name__ == "__main__":
    unittest.main()
