from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from market_fixtures import ohlcv

from crypto_forecaster.config import Settings
from crypto_forecaster.long_notifications import (
    deliver_long_notifications,
    initial_text,
    target_text,
)
from crypto_forecaster.long_scout import (
    DAY,
    STEP,
    VERSION,
    _publish,
    _read,
    paper_outcome,
    profile,
    root,
    settle_paper,
)
from crypto_forecaster.telegram import TelegramDelivery

NOW = 30 * DAY


class OwnerSender:
    def __init__(self, status="SENT"):
        self.receipts = {}
        self.messages = []
        self.status = status

    def owner_delivery_receipt(self, identifier, _directory):
        return self.receipts.get(identifier)

    def deliver_owner_once(self, *, signal_id, text, state_dir):
        self.messages.append(text)
        if self.status == "SENT":
            self.receipts[signal_id] = {"message_id": 99, "delivered_at_ms": NOW}
        return TelegramDelivery(self.status, 99 if self.status == "SENT" else None)


class LongNotificationTests(unittest.TestCase):
    def test_settlement_keeps_alert_reference_separate_from_paper_entry(self):
        self.record["ownerAlert"] = {
            "deliveredAtMs": NOW,
            "referencePrice": 100,
            "trackingStartMs": NOW + STEP,
            "outcome": {},
        }
        bars = ohlcv(
            close=np.array([103, 103]),
            high=np.array([105, 105]),
            low=np.array([102, 102]),
            volume=np.array([1000, 1000]),
            start_ms=NOW + STEP,
            step_ms=STEP,
        )
        bars["open"] = 103
        with patch(
            "crypto_forecaster.long_scout.update_market_cache", return_value=bars
        ):
            settle_paper(
                self.settings,
                self.state,
                None,
                datetime.fromtimestamp((NOW + 3 * STEP) / 1000, UTC),
            )
        self.assertEqual(self.record["outcome"]["entryPrice"], 103)
        self.assertEqual(self.record["ownerAlert"]["outcome"]["entryPrice"], 100)
        self.assertTrue(self.record["ownerAlert"]["outcome"]["targets"]["5"]["hit"])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.settings = Settings(
            scalp_state_dir=directory / "scalp",
            telegram_state_dir=directory / "telegram",
            long_scout_alerts_enabled=True,
        )
        profile_id, _ = profile(self.settings)
        self.record = {
            "id": "a" * 64,
            "symbol": "AAAUSDT",
            "strategy": "LB1",
            "stage": "TEYİT",
            "recordedAtMs": NOW - 10_000,
            "sourceCloseMs": NOW - 60_001,
            "score": 78,
            "referencePrice": 100,
            "stopPct": 3,
            "costBps": 12,
            "reason": "7 günlük zirve üstünde hacimli kapanış",
            "entryOpenMs": NOW + STEP,
            "profileId": profile_id,
            "outcome": {},
        }
        self.state = {
            "version": VERSION,
            "records": [self.record],
            "scannedAtMs": NOW - 5000,
            "regime": {"state": "BULL_CONFIRMED", "dataHealthy": True},
        }
        self.sender = OwnerSender()
        self.addCleanup(self.temporary.cleanup)

    def run_delivery(self, settings=None):
        _publish(self.settings, self.state, NOW)
        return deliver_long_notifications(
            settings or self.settings,
            now=datetime.fromtimestamp(NOW / 1000, UTC),
            notifier=self.sender,
            started_at_ms=NOW - 60_000,
        )

    def test_only_fresh_confirmed_high_score_setups_not_accumulation(self):
        for change in (
            {"stage": "İZLE", "strategy": "LA1"},
            {"score": 60},
            {"recordedAtMs": NOW - DAY},
            {"sourceCloseMs": NOW - DAY},
        ):
            self.state["records"] = [{**self.record, **change}]
            self.run_delivery()
        self.assertFalse(any("LONG ADAYI" in text for text in self.sender.messages))
        self.state["records"] = [self.record]
        self.run_delivery()
        self.assertEqual(sum("LONG ADAYI" in t for t in self.sender.messages), 1)

    def test_current_bull_and_data_health_required(self):
        for regime in (
            {"state": "WAIT", "dataHealthy": True},
            {"state": "BULL_CONFIRMED", "dataHealthy": False},
        ):
            self.state["regime"] = regime
            self.run_delivery()
        self.assertFalse(any("LONG ADAYI" in t for t in self.sender.messages))

    def test_disabled_and_standby_never_send(self):
        self.run_delivery(replace(self.settings, long_scout_alerts_enabled=False))
        with patch(
            "crypto_forecaster.long_notifications.is_primary", return_value=False
        ):
            self.run_delivery()
        self.assertEqual(self.sender.messages, [])

    def test_coin_cooldown_crosses_strategy_and_delivery_retries_deduplicate(self):
        self.state["records"].append({**self.record, "id": "b" * 64, "strategy": "LP1"})
        self.run_delivery()
        self.state = _read(root(self.settings) / "state.json")
        self.run_delivery()
        self.assertEqual(sum("LONG ADAYI" in t for t in self.sender.messages), 1)
        # Owner identity and recipient lists are never exported publicly.
        public = json.loads((root(self.settings) / "public-summary.json").read_text())
        self.assertNotIn("ownerAlert", public["tracked"][0])

    def test_unsent_or_uncertain_initial_never_produces_target_messages(self):
        self.sender.status = "UNCERTAIN"
        self.record["ownerAlert"] = {
            "deliveredAtMs": NOW,
            "referencePrice": 100,
            "trackingStartMs": NOW + STEP,
            "outcome": self.hits(),
        }
        self.run_delivery()
        self.assertFalse(any("DOKUNDU" in t for t in self.sender.messages))

    def hits(self):
        return {
            "complete": True,
            "targets": {
                str(k): {
                    "hit": True,
                    "touchAtMs": NOW + 2 * STEP - 1,
                    "firstBeforeStop": k == 2,
                    "sameBarAmbiguous": k == 3,
                }
                for k in (2, 3, 5, 10, 20)
            },
        }

    def test_percentage_targets_only_once_after_real_initial_delivery(self):
        self.run_delivery()
        self.state = _read(root(self.settings) / "state.json")
        self.state["records"][0]["ownerAlert"]["outcome"] = self.hits()
        self.run_delivery()
        self.run_delivery()
        messages = [t for t in self.sender.messages if "DOKUNDU" in t]
        self.assertEqual(len(messages), 5)
        self.assertIn("Stop sonrası", messages[2])
        self.assertIn("aynı mumda", messages[1])
        self.assertTrue(all(len(t) < 1200 for t in self.sender.messages))

    def test_changed_owner_without_initial_receipt_never_gets_old_targets(self):
        self.run_delivery()
        self.state = _read(root(self.settings) / "state.json")
        self.state["records"][0]["ownerAlert"]["outcome"] = self.hits()
        self.state["records"][0]["recordedAtMs"] = NOW - DAY
        self.sender = OwnerSender()
        self.run_delivery()
        self.assertFalse(any("DOKUNDU" in t for t in self.sender.messages))

    def test_pre_delivery_or_incomplete_bars_do_not_notify(self):
        self.run_delivery()
        self.state = _read(root(self.settings) / "state.json")
        outcome = self.hits()
        for target in outcome["targets"].values():
            target["touchAtMs"] = NOW + STEP - 1
        self.state["records"][0]["ownerAlert"]["outcome"] = outcome
        self.run_delivery()
        outcome["complete"] = False
        self.run_delivery()
        self.assertFalse(any("DOKUNDU" in t for t in self.sender.messages))

    def test_reported_reference_price_not_next_open_is_alert_target_basis(self):
        bars = ohlcv(
            close=np.array([103, 103]),
            high=np.array([105, 105]),
            low=np.array([102, 102]),
            volume=np.array([1000, 1000]),
            start_ms=NOW + STEP,
            step_ms=STEP,
        )
        bars["open"] = 103
        record = {**self.record, "referenceEntryPrice": 100}
        result = paper_outcome(record, bars, NOW + 3 * STEP)
        self.assertEqual(result["entryPrice"], 100)
        self.assertTrue(result["targets"]["5"]["hit"])
        self.assertEqual(
            paper_outcome(self.record, bars, NOW + 3 * STEP)["entryPrice"], 103
        )
        self.assertIn("%2 $102", initial_text(self.record))
        self.record["ownerAlert"] = {"referencePrice": 100}
        self.assertIn("hedef $105", target_text(self.record, 5, result["targets"]["5"]))
