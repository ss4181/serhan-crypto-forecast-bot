from __future__ import annotations

import unittest

from crypto_forecaster.measurement import deadline_ms, measurement_summary

NOW = 1_800_000_000_000
HOUR = 3_600_000


def record(**overrides):
    return dict(dict(kind="scalp-target", signalId="a", targetPercent=2,
                     horizonHours=24, sourceTimeMs=NOW - 25 * HOUR,
                     success=True, notified=True), **overrides)


def cohort(rows, audience="notified", **options):
    return measurement_summary(rows, now_ms=NOW, **options)["audiences"][audience][0]


class MeasurementTests(unittest.TestCase):
    def test_early_hits_do_not_inflate_the_completed_cohort(self):
        rows = [record(success=False), record(sourceTimeMs=NOW - HOUR)]
        c = cohort(rows)
        self.assertEqual(c["hits"], 0)
        self.assertEqual(c["resolvedCount"], 1)
        self.assertEqual(c["hitRate"], 0)
        self.assertEqual(c["openCount"], 1)
        self.assertEqual(c["earlyHits"], 1)
        later = measurement_summary(rows, now_ms=NOW + 24 * HOUR)["audiences"]["notified"][0]
        self.assertEqual(later["hitRate"], .5)

    def test_deadline_boundary_is_inclusive(self):
        self.assertEqual(cohort([record(sourceTimeMs=NOW - 24 * HOUR)])["maturedCount"], 1)
        self.assertEqual(cohort([record(sourceTimeMs=NOW - 24 * HOUR + 1)])["openCount"], 1)

    def test_missing_mature_outcome_suppresses_rate_not_counted_as_failure(self):
        c = cohort([record(), record(success=None)])
        self.assertEqual(c["unresolvedCount"], 1)
        self.assertEqual(c["misses"], 0)
        self.assertEqual(c["resolvedCount"], 1)
        self.assertIsNone(c["hitRate"])

    def test_missing_or_invalid_horizon_is_never_invented(self):
        for value in (None, 0, -1, float("nan"), float("inf"), 1e308, True):
            with self.subTest(horizon=value):
                row = record(horizonHours=value)
                self.assertIsNone(deadline_ms(row))
                c = cohort([row])
                self.assertEqual(c["unknownDeadlineCount"], 1)
                self.assertIsNone(c["hitRate"])

    def test_levels_horizons_and_audiences_are_not_pooled(self):
        rows = [record(), record(targetPercent=3), record(horizonHours=1), record(notified=False, success=False)]
        result = measurement_summary(rows, now_ms=NOW)["audiences"]
        self.assertEqual(len(result["notified"]), 3)
        self.assertEqual(result["muted"][0]["hitRate"], 0)
        self.assertEqual(result["notified"][1]["hitRate"], 1)
        self.assertEqual(result["all"][1]["hitRate"], .5)

    def test_target_before_stop_and_positive_net_are_different(self):
        rows = [record(kind="scalp-bracket", status="TARGET", netBps=-5, horizonHours=1),
                record(kind="scalp-bracket", status="TIME_EXIT", netBps=15, horizonHours=1)]
        c = cohort(rows)
        self.assertEqual(c["hits"], 1)
        self.assertEqual(c["timeExits"], 1)
        self.assertEqual(c["positiveNetCount"], 1)
        self.assertEqual(c["meanNetBps"], 5)
        self.assertEqual(c["positiveNetRate"], .5)

    def test_early_bracket_resolution_excluded_until_its_horizon_ends(self):
        c = cohort([record(kind="scalp-bracket", status="TARGET", horizonHours=1,
                           sourceTimeMs=NOW - HOUR // 2, netBps=80)])
        self.assertEqual(c["openCount"], 1)
        self.assertEqual(c["resolvedCount"], 0)
        self.assertIsNone(c["meanNetBps"])

    def test_missing_net_is_not_zero_and_truncated_history_has_no_rate(self):
        c = cohort([record(kind="scalp-bracket", status="TARGET", horizonHours=1)])
        self.assertEqual(c["hitRate"], 1)
        self.assertEqual(c["missingNetCount"], 1)
        self.assertIsNone(c["meanNetBps"])
        self.assertIsNone(c["positiveNetRate"])
        self.assertIsNone(cohort([record()], history_complete=False)["hitRate"])

    def test_empty_history_has_empty_cohorts(self):
        result = measurement_summary([], now_ms=NOW)
        self.assertEqual(result["audiences"], {"all": [], "notified": [], "muted": []})
