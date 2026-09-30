import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "touch_study", Path(__file__).parents[1] / "research" / "target_touch_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class TouchStudyTests(unittest.TestCase):
    def setUp(self):
        self.stamp = study.STEP - 1
        self.setup = {"time_ms": self.stamp, "reference_price": 100, "cost_bps": 12}
        opens = np.arange(289) * study.STEP
        self.candles = np.column_stack(
            [
                opens,
                np.full(289, 100.5),
                np.full(289, 99.5),
                np.full(289, 100),
                opens + study.STEP - 1,
            ]
        )

    def test_excludes_source_candle_and_future_candles(self):
        self.candles[0, 1] = 110
        self.candles[288, 1] = 110
        result = study.label_setup(self.setup, self.candles, 2 * study.STEP)
        self.assertTrue(result["complete"])
        self.assertFalse(result["mature"])
        self.assertIsNone(result["sides"]["LONG"]["touches"]["2"]["hit"])

    def test_early_hit_is_known_but_miss_is_not_known_before_maturity(self):
        self.candles[1, 1] = 103
        result = study.label_setup(self.setup, self.candles, 2 * study.STEP)
        self.assertIs(result["sides"]["LONG"]["touches"]["3"]["hit"], True)
        self.assertIsNone(result["sides"]["LONG"]["touches"]["5"]["hit"])
        self.assertIsNone(result["sides"]["SHORT"]["touches"]["2"]["hit"])

    def test_mature_side_labels_can_both_hit_without_selecting_winner_side(self):
        self.candles[1, 1:3] = [106, 94]
        result = study.label_setup(self.setup, self.candles, self.stamp + study.DAY + 1)
        for side in study.SIDES:
            self.assertIs(result["sides"][side]["touches"]["5"]["hit"], True)
        self.assertTrue(result["mature"])

    def test_target_touch_after_stop_is_not_a_first_target_win(self):
        self.candles[1, 2] = 98.5
        self.candles[2, 1] = 102.5
        result = study.label_setup(self.setup, self.candles, self.stamp + study.DAY + 1)
        target = result["sides"]["LONG"]["touches"]["2"]
        self.assertIs(target["hit"], True)
        self.assertIs(target["target_before_1pct_stop"], False)
        self.assertGreaterEqual(target["mae_through_touch_bar_pct"], 1.5)

    def test_same_candle_target_and_stop_order_is_unknown(self):
        self.candles[1, 1:3] = [103, 98]
        result = study.label_setup(self.setup, self.candles, self.stamp + study.DAY + 1)
        target = result["sides"]["LONG"]["touches"]["2"]
        self.assertTrue(target["same_bar_target_stop_ambiguous"])
        self.assertIsNone(target["target_before_1pct_stop"])

    def test_future_features_do_not_change_training_imputation_or_scaling(self):
        x = np.array([[1, np.nan], [3, 5]], dtype=float)
        median, scaler = study.preprocess_fit(x)
        before = (median.copy(), scaler.mean.copy(), scaler.scale.copy())
        study.transform(np.array([[1e12, -1e12]]), median, scaler)
        np.testing.assert_array_equal(median, before[0])
        np.testing.assert_array_equal(scaler.mean, before[1])
        np.testing.assert_array_equal(scaler.scale, before[2])

    def test_reconstructs_signal_close_instead_of_using_next_open_as_reference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with gzip.open(root / "BTCUSDT.json.gz", "wt", encoding="utf-8") as handle:
                json.dump({"candles": self.candles.tolist()}, handle)
            setup = {
                **self.setup,
                "symbol": "BTCUSDT",
                "reference_price": None,
                "ledger_next_open_price": 105,
            }
            result = study.attach_labels([setup], root, self.stamp + study.DAY + 1)[0]
            self.assertEqual(result["reference_price"], 100)
            self.assertEqual(result["ledger_next_open_price"], 105)
            self.assertTrue(result["outcome"]["complete"])
            self.assertIs(
                result["outcome"]["sides"]["SHORT"]["touches"]["2"]["hit"], False
            )

    def test_gap_and_reference_mismatch_are_unknown_not_losing_labels(self):
        for candles in (np.delete(self.candles, 10, axis=0), self.candles.copy()):
            if len(candles) == 289:
                candles[0, 3] = 101
            result = study.label_setup(self.setup, candles, self.stamp + study.DAY + 1)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["sides"]["LONG"]["touches"]["2"]["hit"])

    def test_purges_outcome_windows_on_both_boundaries(self):
        rows = [
            {"time_ms": t, "outcome": {"complete": True, "mature": True}}
            for t in range(20 * study.DAY, 30 * study.DAY, study.DAY)
        ]
        train, cal = study.split_indices(rows, 30 * study.DAY)
        self.assertTrue(
            all(rows[i]["time_ms"] + study.DAY < 24 * study.DAY for i in train)
        )
        self.assertTrue(
            all(
                24 * study.DAY <= rows[i]["time_ms"]
                and rows[i]["time_ms"] + study.DAY < 30 * study.DAY
                for i in cal
            )
        )
        self.assertFalse(set(train) & set(cal))

    def test_predictions_need_no_test_outcomes_and_targets_remain_nested(self):
        rng = np.random.default_rng(123)
        rows = [
            {
                "symbol": "BTCUSDT",
                "time_ms": i * study.STEP,
                "features": rng.normal(size=len(study.FEATURES)).tolist(),
                "outcome": {
                    "sides": {
                        side: {
                            "touches": {
                                str(level): {"hit": bool(i % 2)}
                                for level in study.LEVELS
                            }
                        }
                        for side in study.SIDES
                    }
                },
            }
            for i in range(211)
        ]
        # A prediction-time record need not have any future outcome at all.
        del rows[-1]["outcome"]
        predictions, info = study.predict_block(
            rows, list(range(150)), list(range(150, 210)), [210]
        )
        self.assertEqual(info["status"], "fit")
        self.assertEqual(len(predictions), 1)
        for side in study.SIDES:
            p = predictions[0]["probabilities"][side]
            self.assertTrue(0 <= p["5"] <= p["3"] <= p["2"] <= 1)

    def test_feature_allowlist_and_deduplication_ignore_outcome_values(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "ledger.jsonl"
            rows = [
                {
                    "perpetual_symbol": "BTCUSDT",
                    "bar_close_time_ms": self.stamp,
                    "family": f,
                    "horizon_minutes": h,
                    "alert_tier": "KURULUM",
                    "entry_price": 100,
                    "score": 2,
                    "gross_bps": 9999,
                    "net_bps": 8888,
                    "exit_price": 999,
                    "exit_time_ms": 123,
                    "regime_state": "BULL",
                }
                for f in ("B1", "F3")
                for h in (15, 30, 60)
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            setups, malformed = study.load_setups(path)
            self.assertEqual((len(setups), malformed), (1, 0))
            self.assertNotIn(9999, setups[0]["features"])
            self.assertNotIn(8888, setups[0]["features"])
            self.assertNotIn("exit_time_ms", setups[0])
            self.assertEqual(len(setups[0]["features"]), len(study.FEATURES))
