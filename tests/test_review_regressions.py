"""Regression cases discovered during the September project review."""
from __future__ import annotations

import os
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from crypto_forecaster.data import MarketDataError, _validate_frame
from crypto_forecaster.scalping import (
    _first_touch_bracket, _time_exit_outcomes, scalp_setup_assessment,
    _append_outcomes_once, record_scalp_observations, settle_scalp_observations,
    load_scalp_ledger, ScalpScanReport,
    record_scalp_target_setups, settle_scalp_target_outcomes, load_scalp_target_ledger,
    record_scalp_bracket_setups, settle_scalp_bracket_outcomes, load_scalp_bracket_ledger,
)
from crypto_forecaster.universe import load_trade1_universe
from crypto_forecaster.telegram import TelegramNotifier, TelegramRejected
from test_scalping import START_MS, STEP_MS, market_frame, observation
from test_telegram import DIRECT_CREDENTIALS
from datetime import datetime, timezone


class ReviewRegressions(unittest.TestCase):
    def test_targets_and_brackets_survive_failed_append_and_acknowledgement_replay(self):
        manifest = load_trade1_universe()
        items = tuple(replace(observation(family=f), alert_tier="KURULUM", regime_state="BULL") for f in ("B1", "F3"))
        report = ScalpScanReport(manifest.version, 1, 1, 0, (), items, START_MS)
        rows = [dict(family=f, horizon_minutes=h, gross_bps=-40, net_bps=-52,
                     perpetual_symbol="BTCUSDT", regime_state="BULL")
                for f in ("B1", "F3") for h in (15, 30, 60)]
        frame = market_frame()
        after = frame.open_time_ms > items[0].bar_open_time_ms
        frame.loc[after, "low"] = 90
        now = datetime.fromtimestamp((items[0].bar_close_time_ms + 3600000) / 1000, timezone.utc)
        for name, register, settle, load, expected in (
            ("target", record_scalp_target_setups, settle_scalp_target_outcomes, load_scalp_target_ledger, 3),
            ("bracket", record_scalp_bracket_setups, settle_scalp_bracket_outcomes, load_scalp_bracket_ledger, 1),
        ):
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertEqual(register(root, report, manifest=manifest, top_k=1, ledger=rows), 1)
                pending = next((root / f"{name}_pending").glob("*.json"))
                original = pending.read_text(encoding="utf-8")
                with patch("crypto_forecaster.scalping.load_cache", return_value=frame):
                    with patch(f"crypto_forecaster.scalping._append_scalp_{name}_ledger", side_effect=OSError("disk full")):
                        with self.assertRaises(OSError):
                            settle(root, root, now=now)
                    self.assertEqual(pending.read_text(encoding="utf-8"), original)
                    settle(root, root, now=now)
                    self.assertEqual(len(load(root)), expected)
                    pending.write_text(original, encoding="utf-8")
                    settle(root, root, now=now)
                    self.assertEqual(len(load(root)), expected)

    def test_ledger_append_failure_preserves_pending_work_and_replay_is_unique(self):
        item = observation()
        manifest = load_trade1_universe()
        report = ScalpScanReport(manifest.version, 1, 1, 0, (), (item,), START_MS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_scalp_observations(root, report.observations, manifest=manifest)
            pending = next((root / "pending").glob("*.json"))
            original = pending.read_text(encoding="utf-8")
            now = datetime.fromtimestamp((item.bar_close_time_ms + 3600000) / 1000, timezone.utc)
            with patch("crypto_forecaster.scalping.load_cache", return_value=market_frame()):
                with patch("crypto_forecaster.scalping._append_scalp_ledger", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        settle_scalp_observations(root, root, now=now)
                self.assertEqual(pending.read_text(encoding="utf-8"), original)
                settle_scalp_observations(root, root, now=now)
                self.assertEqual(len(load_scalp_ledger(root)), 3)
                # Simulate a restart after durable append but before acknowledgement.
                pending.write_text(original, encoding="utf-8")
                settle_scalp_observations(root, root, now=now)
                self.assertEqual(len(load_scalp_ledger(root)), 3)

    def test_append_recovers_after_a_truncated_tail_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            path.write_text('{"id": "old"}\n{"id": "brok', encoding="utf-8")
            _append_outcomes_once(path, [{"id": "old"}, {"id": "new"}], ("id",))
            _append_outcomes_once(path, [{"id": "new"}], ("id",))
            valid = []
            for line in path.read_text().splitlines():
                try:
                    valid.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.assertEqual(valid, [{"id": "old"}, {"id": "new"}])

    @patch.dict(os.environ, DIRECT_CREDENTIALS)
    def test_partial_delivery_retries_only_failed_recipient(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            notifier = TelegramNotifier(state_dir=state)
            with patch.object(notifier, "_direct_recipients", return_value=(500100, 700200)), patch.object(
                notifier, "_send_to_chat", side_effect=[99, TelegramRejected("403"), 100]
            ) as sender:
                first = notifier.deliver_once(signal_id="a" * 64, text="test", state_dir=state)
                second = notifier.deliver_once(signal_id="a" * 64, text="test", state_dir=state)
                self.assertEqual(first.status, "PARTIAL")
                self.assertEqual(second.status, "SENT")
                self.assertEqual([c.args[0] for c in sender.call_args_list], [500100, 700200, 700200])

    @patch.dict(os.environ, DIRECT_CREDENTIALS)
    def test_new_member_does_not_receive_replayed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            notifier = TelegramNotifier(state_dir=state)
            with patch.object(notifier, "_direct_recipients", return_value=(500100,)) as recipients, patch.object(
                notifier, "_send_to_chat", return_value=99
            ) as sender:
                notifier.deliver_once(signal_id="b" * 64, text="test", state_dir=state)
                recipients.return_value = (500100, 700200)
                result = notifier.deliver_once(signal_id="b" * 64, text="test", state_dir=state)
                self.assertEqual(result.status, "DEDUPLICATED")
                self.assertEqual(sender.call_count, 1)

    @patch.dict(os.environ, DIRECT_CREDENTIALS)
    def test_revocation_prevents_retry_to_removed_member(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            notifier = TelegramNotifier(state_dir=state)
            with patch.object(notifier, "_direct_recipients", return_value=(500100, 700200)) as recipients, patch.object(
                notifier, "_send_to_chat", side_effect=[99, TelegramRejected("403")]
            ) as sender:
                notifier.deliver_once(signal_id="c" * 64, text="test", state_dir=state)
                recipients.return_value = (500100,)
                result = notifier.deliver_once(signal_id="c" * 64, text="test", state_dir=state)
                self.assertEqual(result.status, "DEDUPLICATED")
                self.assertEqual(sender.call_count, 2)

    def test_nan_and_infinity_market_values_are_rejected(self):
        for column in ("close", "high", "low", "volume", "taker_buy_base"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(column=column, value=value):
                    frame = market_frame(3)
                    frame.loc[1, column] = value
                    with self.assertRaises(MarketDataError):
                        _validate_frame(frame)

    def test_assessment_counts_a_market_move_once_across_families(self):
        items = tuple(replace(observation(family=f), regime_state="BULL") for f in ("B1", "F3"))
        rows = [dict(family=f, perpetual_symbol="BTCUSDT", regime_state="BULL",
                     horizon_minutes=h, gross_bps=-40, net_bps=-52,
                     round_trip_cost_bps=12, score=2,
                     bar_close_time_ms=START_MS + i * 86400000,
                     exit_time_ms=START_MS + i * 86400000 + h * 60000)
                for f in ("B1", "F3") for h in (15, 30, 60) for i in range(12)]
        result = scalp_setup_assessment(items, rows)
        duplicate_result = scalp_setup_assessment(items, rows + rows)
        self.assertEqual(result.sample_count, 12)
        self.assertEqual(result, duplicate_result)
        self.assertAlmostEqual(result.success_probability, 13 / 14)

    def test_first_touch_requires_contiguous_data_from_entry(self):
        record = dict(bar_close_time_ms=START_MS, direction="YUKARI",
                      horizon_minutes=15, target_bps=100, stop_bps=100,
                      round_trip_cost_bps=12)
        window = pd.DataFrame([
            [START_MS + STEP_MS, 100, 100.5, 99.5, 100],
            [START_MS + 2 * STEP_MS, 100, 100.5, 99.5, 100],
            [START_MS + 3 * STEP_MS, 100, 102, 99.5, 101],
        ], columns=["close_time_ms", "open", "high", "low", "close"])
        for missing in (0, 1):
            self.assertIsNone(_first_touch_bracket(record, window.drop(missing), deadline_reached=True))
        self.assertIsNone(_first_touch_bracket(record, window.iloc[:2], deadline_reached=True))
        self.assertEqual(_first_touch_bracket(record, window, deadline_reached=True)["resolution"], "TARGET")
        window.loc[0, "high"] = 102
        self.assertEqual(_first_touch_bracket(record, window.drop(1), deadline_reached=True)["resolution"], "TARGET")

    def test_time_exit_does_not_shift_horizon_past_a_missing_bar(self):
        item = observation()
        record = dict(bar_open_time_ms=item.bar_open_time_ms,
                      bar_close_time_ms=item.bar_close_time_ms,
                      horizons_minutes=[15, 30, 60], round_trip_cost_bps=12)
        frame = market_frame()
        frame = frame.drop(frame.index[frame.open_time_ms == item.bar_open_time_ms + STEP_MS]).reset_index(drop=True)
        self.assertIsNone(_time_exit_outcomes(record, frame))
