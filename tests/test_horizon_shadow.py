import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_scalping import bull_regime, market_frame, observation

from crypto_forecaster.config import Settings
from crypto_forecaster.dashboard import build_dashboard_payload
from crypto_forecaster.horizon_shadow import (
    VERSION,
    _metric,
    format_horizon_shadow_status,
    load_horizon_shadow_summary,
    run_horizon_shadow,
)
from crypto_forecaster.scalping import (
    STRATEGY_HORIZON_DIRECTION_POLICY as POLICY,
)
from crypto_forecaster.scalping import (
    ScalpScanReport,
    scalp_cache_path,
    scalp_setup_assessment,
    scalp_setup_direction,
)
from crypto_forecaster.universe import load_trade1_universe

STEP = 300_000
OPEN = 1_780_000_000_000 // STEP * STEP
CLOSE = OPEN + STEP - 1


def items():
    return tuple(
        replace(
            observation(family=f, score=6),
            price=100.0,
            bar_open_time_ms=OPEN,
            bar_close_time_ms=CLOSE,
            alert_tier="KURULUM",
            regime_state="BULL",
            execution_eligible=True,
            estimated_cost_bps=12.0,
            spread_bps=1.0,
            quote_volume_24h_usdt=10_000_000,
            funding_rate_bps=1.0,
            volatility_bps=25.0,
        )
        for f in ("B1", "F3")
    )


def history(gross=None):
    gross = gross or {15: -40.0, 30: 8.0, 60: 70.0}
    return [
        {
            "family": f,
            "perpetual_symbol": "BTCUSDT",
            "regime_state": "BULL",
            "horizon_minutes": h,
            "gross_bps": gross[h],
            "net_bps": gross[h] - 12.0,
            "round_trip_cost_bps": 12.0,
            "score": 2.0,
            "bar_close_time_ms": CLOSE - 3_600_000 - i * STEP,
            "exit_time_ms": CLOSE - 3_600_000 - i * STEP + h * 60_000,
        }
        for f in ("B1", "F3")
        for i in range(1, 13)
        for h in (15, 30, 60)
    ]


def report():
    return ScalpScanReport(
        load_trade1_universe().version,
        89,
        89,
        0,
        (),
        items(),
        CLOSE + 20_000,
        regime=bull_regime(),
        quoted=89,
        newest_close_time_ms=CLOSE,
    )


def moment(value):
    return datetime.fromtimestamp(value / 1000, UTC)


class FixedHorizonTests(unittest.TestCase):
    def test_long_uses_fixed_60m_despite_other_horizons(self):
        self.assertNotEqual(scalp_setup_direction(items(), history())[0], "YUKARI")
        assessment = scalp_setup_assessment(items(), history(), direction_policy=POLICY)
        self.assertEqual(
            (assessment.direction, assessment.horizon_minutes), ("YUKARI", 60)
        )
        self.assertEqual(assessment.strategy_code, "BULL_CONTINUATION_LONG")

    def test_short_uses_30m_without_choosing_best_return_horizon(self):
        rows = history({15: -500, 30: -40, 60: 8})
        a = scalp_setup_assessment(items(), rows, direction_policy=POLICY)
        self.assertEqual((a.direction, a.horizon_minutes), ("AŞAĞI", 30))
        self.assertAlmostEqual(a.expected_net_bps, 28)

    def test_opposing_playbooks_and_family_conflicts_stay_silent(self):
        self.assertEqual(
            scalp_setup_direction(
                items(), history({15: 8, 30: -40, 60: 70}), policy=POLICY
            )[0],
            "KARIŞIK",
        )
        rows = history()
        for row in rows:
            if row["family"] == "F3" and row["horizon_minutes"] == 60:
                row.update(gross_bps=-40, net_bps=-52)
        self.assertNotIn(
            scalp_setup_direction(items(), rows, policy=POLICY)[0], ("YUKARI", "AŞAĞI")
        )

    def test_missing_primary_horizon_cannot_fall_back_to_another_one(self):
        rows = [
            r for r in history({15: 70, 30: 70, 60: 70}) if r["horizon_minutes"] != 60
        ]
        self.assertNotIn(
            scalp_setup_direction(items(), rows, policy=POLICY)[0], ("YUKARI", "AŞAĞI")
        )
        with self.assertRaises(ValueError):
            scalp_setup_direction(items(), rows, policy="typo")


class ShadowLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.settings = Settings(
            scalp_state_dir=root / "scalp",
            scalp_data_dir=root / "data",
            outcome_state_dir=root / "normal",
            scalp_minimum_alert_score=5.0,
            scalp_horizon_shadow_enabled=True,
        )
        self.manifest = load_trade1_universe()

    def run_scan(self, scan=None, rows=None, settings=None):
        scan = scan or report()
        return run_horizon_shadow(
            settings or self.settings,
            scan,
            manifest=self.manifest,
            ledger=history() if rows is None else rows,
            now=moment(scan.evaluated_at_ms),
        )

    def test_prospective_isolated_silent_and_no_duplicate_after_restart(self):
        future = [
            dict(r, exit_time_ms=CLOSE + 86_400_000, gross_bps=-200, net_bps=-212)
            for r in history()
        ] * 3
        with patch("crypto_forecaster.scalping.TelegramNotifier") as notifier:
            summary = self.run_scan(rows=history() + future)
            notifier.assert_not_called()
        self.assertEqual(summary["arms"]["baseline"]["candidates"], 0)
        self.assertEqual(summary["arms"]["proposed"]["candidates"], 1)
        self.assertEqual(summary["arms"]["additional"]["candidates"], 1)
        self.assertFalse((self.settings.scalp_state_dir / "target_pending").exists())
        pending = list(
            self.settings.scalp_state_dir.glob("experiments/**/target_pending/*.json")
        )
        self.assertEqual(len(pending), 1)
        payload = json.loads(pending[0].read_text(encoding="utf-8"))
        self.assertIs(payload["notification_sent"], False)
        self.assertEqual(payload["assessment_horizon_minutes"], 60)
        self.assertEqual(payload["direction"], "YUKARI")
        # A restart on the same candle cannot revise its decision with new data.
        second = self.run_scan(rows=history({15: -40, 30: -40, 60: -40}))
        self.assertEqual(second["arms"]["proposed"]["candidates"], 1)
        self.assertEqual(json.loads(pending[0].read_text(encoding="utf-8")), payload)
        self.assertIn(
            "Yeni 1",
            format_horizon_shadow_status(self.settings, now=moment(CLOSE + 20_000)),
        )

    def test_settles_future_targets_and_net_without_counting_early_hits_as_rate(self):
        self.run_scan()
        frame = market_frame(320)
        frame["open_time_ms"] = OPEN + np.arange(len(frame)) * STEP
        frame["close_time_ms"] = frame["open_time_ms"] + STEP - 1
        frame["open"] = 102.0
        frame["close"] = 102.0
        frame["high"] = 102.1
        frame["low"] = 101.9
        frame.loc[0, ["open", "close", "high", "low"]] = [100, 100, 100.1, 99.9]
        frame.loc[1, ["open", "close", "high", "low"]] = [100, 102, 106, 99.99]
        path = scalp_cache_path(self.settings.scalp_data_dir, "BTCUSDT")
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)

        def next_scan(minutes):
            stamp = CLOSE + minutes * 60_000 + 20_000
            return replace(
                report(),
                observations=(),
                evaluated_at_ms=stamp,
                newest_close_time_ms=stamp // STEP * STEP - 1,
            )

        early = self.run_scan(next_scan(10))
        self.assertIsNone(early["arms"]["proposed"]["targets"]["2"]["rate"])
        self.assertEqual(early["arms"]["proposed"]["targets"]["2"]["pending"], 1)
        complete = self.run_scan(next_scan(24 * 60))
        arm = complete["arms"]["additional"]
        for level in ("2", "3", "5"):
            self.assertEqual(arm["targets"][level]["rate"], 1)
            self.assertEqual(arm["targets"][level]["resolved"], 1)
            self.assertEqual(len(arm["targets"][level]["wilson95"]), 2)
        self.assertEqual(arm["bracket"]["rate"], 1)
        self.assertGreater(arm["bracket"]["meanNetBps"], 0)
        for h in ("15", "30", "60"):
            self.assertEqual(arm["forward"][h]["rate"], 1)

    def test_unhealthy_feed_and_future_only_history_create_no_candidates(self):
        unhealthy = replace(report(), fresh=40)
        self.assertEqual(self.run_scan(unhealthy)["arms"]["proposed"]["candidates"], 0)
        future = [dict(row, exit_time_ms=CLOSE + 86_400_000) for row in history()]
        self.assertEqual(
            self.run_scan(rows=future)["arms"]["proposed"]["candidates"], 0
        )

    def test_changed_thresholds_start_new_profile_without_erasing_old_evidence(self):
        first = self.run_scan()
        second = self.run_scan(
            settings=replace(self.settings, scalp_minimum_alert_score=9)
        )
        self.assertNotEqual(first["profileId"], second["profileId"])
        self.assertEqual(second["arms"]["proposed"]["candidates"], 0)
        self.assertEqual(
            len(list(self.settings.scalp_state_dir.glob("experiments/**/config.json"))),
            2,
        )

    def test_missing_results_are_not_failures_or_zero_success_rates(self):
        key = ("BTCUSDT", CLOSE, "YUKARI")
        metric = _metric(
            [],
            {key: {}},
            now_ms=CLOSE + 7_200_000,
            maturity_ms=3_600_000,
            field="success",
        )
        self.assertEqual(metric["missing"], 1)
        self.assertEqual(metric["resolved"], 0)
        self.assertIsNone(metric["rate"])

    def test_dashboard_exports_only_allowlisted_comparison_fields(self):
        self.run_scan()
        path = self.settings.scalp_state_dir / "experiments" / VERSION / "summary.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        source["token"] = "never-export-this"
        source["arms"]["proposed"]["recipient"] = "never-export-this"
        path.write_text(json.dumps(source), encoding="utf-8")
        public = build_dashboard_payload(self.settings, now=moment(CLOSE + 20_000))
        self.assertEqual(public["horizonShadow"]["arms"]["proposed"]["candidates"], 1)
        self.assertNotIn("never-export-this", json.dumps(public))
        source["arms"] = []
        path.write_text(json.dumps(source), encoding="utf-8")
        self.assertIsNone(
            load_horizon_shadow_summary(self.settings)["arms"]["proposed"]["candidates"]
        )

    def test_restricted_export_reads_only_public_aggregates(self):
        self.run_scan()
        root = self.settings.scalp_state_dir / "experiments" / VERSION
        private = root / "summary.json"
        public = root / "public-summary.json"
        payload = json.loads(public.read_text(encoding="utf-8"))
        self.assertNotIn("profileId", payload)
        self.assertNotIn("lastScan", payload)
        self.assertNotIn("filters", payload)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(public.stat().st_mode), 0o644)
        original_read = Path.read_text

        def restricted_read(path, *args, **kwargs):
            if path == private:
                raise PermissionError("private research state")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", restricted_read):
            exported = build_dashboard_payload(self.settings)["horizonShadow"]
        self.assertEqual(exported["arms"]["proposed"]["candidates"], 1)
        self.assertTrue(exported["dataHealthy"])
