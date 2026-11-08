from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from oi_breakout_study import oi_confirms


class OIBreakoutTests(unittest.TestCase):
    def test_joint_support_and_minimum_change(self):
        self.assertTrue(oi_confirms(.005, 0))
        self.assertTrue(oi_confirms(.001, .007))
        self.assertFalse(oi_confirms(.004, .004))
        self.assertFalse(oi_confirms(.01, -.0001))

    def test_unknown_never_confirms(self):
        for unknown in (float("nan"), float("inf"), -float("inf")):
            self.assertFalse(oi_confirms(unknown, .01))
            self.assertFalse(oi_confirms(.01, unknown))


if __name__ == "__main__":
    unittest.main()
