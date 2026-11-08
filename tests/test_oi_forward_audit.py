"""Synthetic safety tests; not evidence of market performance."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from oi_forward_audit_20261023 import outcome, evaluate_portfolio
from rolling_payoff_study import BAR, HOLD


class ForwardAuditTests(unittest.TestCase):
    def frame(self, n):
        return pd.DataFrame([dict(open_time_ms=i*BAR, close_time_ms=(i+1)*BAR-1,
                                  open=100., high=100., low=100., close=100.) for i in range(n)])

    def test_pending_and_stale_missing_are_not_zero_returns(self):
        f = self.frame(10)
        pending = outcome(f, 0, int(f.close_time_ms.iloc[-1]))
        missing = outcome(f, 0, 1000*BAR)
        self.assertEqual(pending['outcome_status'], 'PENDING')
        self.assertEqual(missing['outcome_status'], 'UNKNOWN_DATA')
        self.assertNotIn('gross_bps', pending)
        self.assertNotIn('gross_bps', missing)

    def test_unresolved_capacity_and_pre_cut_carry_retained(self):
        rows = [dict(symbol=s, signal_ms=0, entry_ms=1, exit_ms=100,
                     outcome_status='UNKNOWN_DATA') for s in ('A', 'B', 'C')]
        rows.append(dict(symbol='D', signal_ms=50, entry_ms=51, exit_ms=60,
                         outcome_status='SETTLED', gross_bps=300., reason='TARGET'))
        result = evaluate_portfolio(rows, cut=20)
        self.assertEqual(result['settled_measurement']['at_30bps']['n'], 0)
        self.assertEqual(len(result['unresolved_pre_cut_carry']), 3)
        self.assertEqual(result['complete_case_reference']['at_30bps']['n'], 1)

    def test_frozen_ambiguous_stop_and_full_window(self):
        f = self.frame(HOLD+1)
        f.loc[1, ['high','low']] = [104., 98.]
        r = outcome(f, 0, int(f.close_time_ms.iloc[-1]))
        self.assertEqual(r['outcome_status'], 'SETTLED')
        self.assertEqual(r['reason'], 'AMBIGUOUS_STOP_FIRST')
        self.assertAlmostEqual(r['gross_bps'], -150.)
        self.assertEqual(outcome(f.iloc[:5], 0, 4*BAR)['outcome_status'], 'PENDING')


if __name__ == '__main__':
    unittest.main()
