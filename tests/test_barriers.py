from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_forecaster.features import (
    BOTH_IN_ONE_CANDLE,
    LOWER_FIRST,
    NO_TOUCH,
    UPPER_FIRST,
    build_supervised_dataset,
)


STEP_MS = 300_000


def quiet_bars(count: int = 400, spike_at: int | None = None, spike: float = 0.0) -> pd.DataFrame:
    """A calm series whose ATR is far below the barrier, plus one optional jump."""
    index = np.arange(count)
    close = 100.0 * (1.0 + 0.0001 * ((-1.0) ** index))
    # Every indicator needs some variation: a perfectly flat candle has no
    # range and a constant volume has no z-score.
    high = close * 1.00005
    low = close * 0.99995
    volume = 10.0 + (index % 7)
    if spike_at is not None:
        if spike > 0:
            high[spike_at] = close[spike_at] * (1.0 + spike)
        else:
            low[spike_at] = close[spike_at] * (1.0 + spike)
    open_time = 1_700_000_000_000 + np.arange(count) * STEP_MS
    return pd.DataFrame(
        {
            "open_time_ms": open_time,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time_ms": open_time + STEP_MS - 1,
        }
    )


def drifting_bars(count: int = 900, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.0015, count)))
    spread = close * rng.uniform(0.0003, 0.0015, count)
    open_time = 1_700_000_000_000 + np.arange(count) * STEP_MS
    return pd.DataFrame(
        {
            "open_time_ms": open_time,
            "open": np.r_[close[0], close[:-1]],
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.lognormal(8, 0.3, count),
            "close_time_ms": open_time + STEP_MS - 1,
        }
    )


def dataset(bars: pd.DataFrame, **kwargs):  # type: ignore[no-untyped-def]
    options = {
        "barrier_atr_multiple": 1.0,
        "barrier_horizon_candles": 12,
        "minimum_barrier_bps": 40.0,
    }
    options.update(kwargs)
    return build_supervised_dataset(bars, **options)


class BarrierTests(unittest.TestCase):
    def test_barrier_never_sits_closer_than_the_cost_floor(self) -> None:
        # A 5m candle's ATR is far smaller than a round trip, so without the
        # floor the model would be paid less than it spends to collect.
        built = dataset(drifting_bars(), minimum_barrier_bps=40.0)
        self.assertTrue(np.all(built.barrier_bps >= 40.0 - 1e-9))

    def test_upper_touch_first_is_labelled_up(self) -> None:
        built = dataset(quiet_bars(spike_at=300, spike=0.02), minimum_barrier_bps=50.0)
        touched = np.nonzero(built.first_touch == UPPER_FIRST)[0]
        self.assertGreater(touched.size, 0)
        self.assertEqual(set(built.y[touched].tolist()), {1.0})
        self.assertNotIn(LOWER_FIRST, set(built.first_touch.tolist()))

    def test_lower_touch_first_is_labelled_down(self) -> None:
        built = dataset(quiet_bars(spike_at=300, spike=-0.02), minimum_barrier_bps=50.0)
        touched = np.nonzero(built.first_touch == LOWER_FIRST)[0]
        self.assertGreater(touched.size, 0)
        self.assertEqual(set(built.y[touched].tolist()), {0.0})
        self.assertNotIn(UPPER_FIRST, set(built.first_touch.tolist()))

    def test_calm_market_times_out_instead_of_touching(self) -> None:
        built = dataset(quiet_bars(), minimum_barrier_bps=50.0)
        self.assertEqual(set(built.first_touch.tolist()), {NO_TOUCH})
        self.assertTrue(np.all(np.abs(built.timeout_return_bps) < built.barrier_bps))

    def test_both_barriers_in_one_candle_is_charged_as_a_loss_either_way(self) -> None:
        built = dataset(drifting_bars(), minimum_barrier_bps=40.0)
        forced = built.first_touch.copy()
        forced[:] = BOTH_IN_ONE_CANDLE
        object.__setattr__(built, "first_touch", forced)
        long_side = built.outcome_bps(np.ones(len(built)), 20.0)
        short_side = built.outcome_bps(-np.ones(len(built)), 20.0)
        # Neither direction may claim the favourable level it cannot prove.
        self.assertTrue(np.all(long_side < 0))
        self.assertTrue(np.all(short_side < 0))
        self.assertTrue(np.allclose(long_side, short_side))

    def test_outcome_is_symmetric_for_a_resolved_barrier(self) -> None:
        built = dataset(drifting_bars(), minimum_barrier_bps=40.0)
        resolved = np.nonzero((built.first_touch == UPPER_FIRST) | (built.first_touch == LOWER_FIRST))[0]
        self.assertGreater(resolved.size, 0)
        side = np.ones(resolved.size)
        long_side = built.outcome_bps(side, 0.0, indices=resolved)
        short_side = built.outcome_bps(-side, 0.0, indices=resolved)
        self.assertTrue(np.allclose(long_side, -short_side))

    def test_winning_trade_clears_its_own_cost(self) -> None:
        built = dataset(drifting_bars(), minimum_barrier_bps=40.0)
        winners = np.nonzero(built.first_touch == UPPER_FIRST)[0]
        self.assertGreater(winners.size, 0)
        net = built.outcome_bps(np.ones(winners.size), 20.0, indices=winners)
        self.assertTrue(np.all(net >= 20.0 - 1e-9))

    def test_label_horizon_reports_how_far_a_label_looks(self) -> None:
        built = dataset(drifting_bars(), barrier_horizon_candles=9, minimum_barrier_bps=200.0)
        self.assertEqual(built.label_horizon, 9)

    def test_unresolvable_tail_is_dropped(self) -> None:
        bars = drifting_bars(count=900)
        long_horizon = dataset(bars, barrier_horizon_candles=30)
        short_horizon = dataset(bars, barrier_horizon_candles=5)
        self.assertEqual(len(short_horizon) - len(long_horizon), 25)

    def test_features_still_ignore_the_future(self) -> None:
        bars = drifting_bars(count=900)
        extended = pd.concat([bars, drifting_bars(count=40, seed=99)], ignore_index=True)
        extended["open_time_ms"] = 1_700_000_000_000 + np.arange(len(extended)) * STEP_MS
        extended["close_time_ms"] = extended["open_time_ms"] + STEP_MS - 1
        base = dataset(bars)
        longer = dataset(extended)
        shared = len(base)
        self.assertTrue(np.allclose(base.x[:shared], longer.x[:shared]))


if __name__ == "__main__":
    unittest.main()
