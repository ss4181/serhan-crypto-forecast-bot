"""Safety checks for the offline research selector, not a live strategy test."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from rolling_payoff_study import BAR, DAY, HOLD, portfolio, resolve, training_rows


class RollingPayoffTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{"open_time_ms": i * BAR, "close_time_ms": (i + 1) * BAR - 1,
                              "open": 100., "high": 100., "low": 100., "close": 100.}
                             for i in range(HOLD + 2)])

    def test_next_open_entry_and_ambiguous_stop_first(self):
        frame = self.frame()
        frame.loc[0, ["high", "low"]] = [104., 95.]
        frame.loc[1, ["high", "low"]] = [103., 98.]
        trade = resolve(frame, 0, 2.)
        self.assertEqual(trade["entry_ms"], BAR)
        self.assertEqual(trade["reason"], "AMBIGUOUS_STOP_FIRST")
        self.assertAlmostEqual(trade["gross_bps"], -150.)

    def test_gaps_and_incomplete_windows_unknown(self):
        frame = self.frame()
        self.assertIsNone(resolve(frame.iloc[:-2], 0, 2.))
        frame.loc[2, "open_time_ms"] += BAR
        self.assertIsNone(resolve(frame, 0, 2.))

    def test_closed_training_only(self):
        cut = 30 * DAY
        rows = [{"symbol": "A", "signal_ms": cut - DAY, "entry_ms": cut - DAY + BAR, "exit_ms": cut - 1},
                {"symbol": "B", "signal_ms": cut - DAY, "entry_ms": cut - DAY + BAR, "exit_ms": cut},
                {"symbol": "C", "signal_ms": cut, "entry_ms": cut + BAR, "exit_ms": cut + DAY},
                {"symbol": "D", "signal_ms": 0, "entry_ms": BAR, "exit_ms": DAY}]
        self.assertEqual([r["symbol"] for r in training_rows(rows, cut)], ["A"])

    def test_capacity_duplicate_and_cross_week_carry(self):
        busy = {"A": 100}
        rows = [{"symbol": s, "entry_ms": 50, "exit_ms": 150} for s in ("D", "C", "B", "A")]
        self.assertEqual([r["symbol"] for r in portfolio(rows, busy)], ["B", "C"])
        self.assertEqual(portfolio([{"symbol": "D", "entry_ms": 75, "exit_ms": 200}], busy), [])
        self.assertEqual(len(portfolio([{"symbol": "D", "entry_ms": 151, "exit_ms": 200}], busy)), 1)

    def test_unsettled_training_trade_still_occupies_slot(self):
        cut = 30 * DAY
        rows = [{"symbol": "A", "signal_ms": cut - DAY, "entry_ms": cut - DAY + BAR,
                 "exit_ms": cut + BAR},
                {"symbol": "A", "signal_ms": cut - 10 * BAR, "entry_ms": cut - 9 * BAR,
                 "exit_ms": cut - BAR}]
        self.assertEqual(training_rows(rows, cut), [])


if __name__ == "__main__":
    unittest.main()
