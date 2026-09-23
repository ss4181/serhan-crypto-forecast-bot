from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.persistence import atomic_write_json


class AtomicJsonTests(unittest.TestCase):
    def test_replace_failure_preserves_previous_state_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending" / "signal.json"
            path.parent.mkdir()
            path.write_text('{"old": true}\n', encoding="utf-8")

            with patch("crypto_forecaster.persistence.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"new": True})

            self.assertEqual(path.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_write_replaces_complete_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending" / "signal.json"
            atomic_write_json(path, {"signal": "SOL", "price": 12.5})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"price": 12.5, "signal": "SOL"}\n',
            )


if __name__ == "__main__":
    unittest.main()
