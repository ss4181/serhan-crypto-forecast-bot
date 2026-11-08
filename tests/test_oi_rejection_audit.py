import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from oi_rejection_audit_20261023 import classify, touch_diagnostic, summarize_events
from oi_forward_audit_20261023 import outcome
from rolling_payoff_study import BAR, HOLD


class RejectionTests(unittest.TestCase):
    def test_filter_order_and_boundary(self):
        r = SimpleNamespace(btc_complete=True, eth_complete=True, btc_return=.01,
                            eth_return=.01, btc_oi=.005, eth_oi=0.)
        self.assertEqual(classify(r, 5_000_000), 'confirmed_before_capacity')
        r.eth_oi = -.001
        self.assertEqual(classify(r, 5_000_000), 'oi_not_confirmed')
        r.eth_return = 0.
        self.assertEqual(classify(r, 5_000_000), 'benchmark_prices_not_rising')
        self.assertEqual(classify(r, 1), 'liquidity_below_5m')
        self.assertEqual(classify(None, 1), 'missing_context')

    def test_later_touch_is_not_a_profitable_trade(self):
        f = pd.DataFrame([dict(open_time_ms=i*BAR, close_time_ms=(i+1)*BAR-1,
                              open=100., high=100., low=100., close=100.) for i in range(HOLD+1)])
        f.loc[1, 'low'] = 98.
        f.loc[2, 'high'] = 106.
        row = {'symbol': 'A', **outcome(f, 0, HOLD*BAR+BAR-1)}
        row.update(touch_diagnostic(f, 0, row))
        pending = {'symbol': 'B', 'outcome_status': 'PENDING', 'eventual_touch_pct': None}
        summary = summarize_events([row, pending])
        self.assertEqual(summary['eventual_touch_counts_not_profit'], {'2': 1, '3': 1, '5': 1})
        m = summary['overlapping_event_measurement_not_portfolio']
        self.assertEqual(m['target_before_stop_count'], 0)
        self.assertEqual(m['at_50bps']['n'], 1)
        self.assertAlmostEqual(m['at_50bps']['mean_net_bps'], -200.)
        self.assertIsNone(touch_diagnostic(f, 0, pending)['eventual_touch_pct'])


if __name__ == '__main__':
    unittest.main()
