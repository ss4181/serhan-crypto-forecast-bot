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


class MeasurementSummaryTests(unittest.TestCase):
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

    def test_cohorts_separate_direction_regime_and_strategy_with_wilson_interval(self):
        records = [
            {
                "kind": "scalp-target",
                "targetPercent": 2.0,
                "horizonHours": 24,
                "direction": direction,
                "regimeState": regime,
                "strategy": strategy,
                "policyVersion": policy,
                "sourceTimeMs": 1_700_000_000_000 + i * 86_400_000,
                "deadlineTimeMs": 1_700_086_400_000 + i * 86_400_000,
                "cohortMatured": True,
                "success": i % 2 == 0,
                "notified": True,
            }
            for i, (direction, regime, strategy, policy) in enumerate(
                (
                    ("YUKARI", "BULL", "A", "v1"),
                    ("AŞAĞI", "TRANSITION", "B", "v1"),
                    ("YUKARI", "BULL", "A", "v2"),
                )
            )
        ]
        result = measurement_summary(records, now_ms=1_800_000_000_000)
        cohorts = result["audiences"]["notified"]
        self.assertEqual(len(cohorts), 3)
        self.assertEqual(
            {
                (row["direction"], row["regime"], row["strategy"], row["policyVersion"])
                for row in cohorts
            },
            {
                ("YUKARI", "BULL", "A", "v1"),
                ("AŞAĞI", "TRANSITION", "B", "v1"),
                ("YUKARI", "BULL", "A", "v2"),
            },
        )
        self.assertTrue(all(len(row["hitRateWilson95"]) == 2 for row in cohorts))
        self.assertTrue(all(row["distinctUtcDays"] == 1 for row in cohorts))

    def test_missing_mature_outcome_disables_rate_and_interval(self):
        result = measurement_summary(
            [{
                "kind": "scalp-bracket",
                "horizonHours": 1,
                "sourceTimeMs": 1_700_000_000_000,
                "deadlineTimeMs": 1_700_003_600_000,
                "cohortMatured": True,
                "success": None,
                "status": "VERİ EKSİK",
                "notified": True,
            }],
            now_ms=1_800_000_000_000,
        )
        cohort = result["audiences"]["notified"][0]
        self.assertIsNone(cohort["hitRate"])
        self.assertIsNone(cohort["hitRateWilson95"])
        self.assertEqual(cohort["unresolvedCount"], 1)

    def test_rolling_windows_keep_realised_rate_and_wilson_interval(self):
        rows = [{
            "kind": "scalp-forward", "symbol": "SOLUSDT", "families": ["B3"],
            "strategy": "B3", "direction": "UNKNOWN", "regimeState": "BULL",
            "policyVersion": "p2", "horizonHours": 0.25,
            "sourceTimeMs": 1_700_000_000_000 + i * 60_000,
            "deadlineTimeMs": 1_700_000_900_000 + i * 60_000,
            "cohortMatured": True, "success": i % 2 == 0,
            "status": "NET POZİTİF" if i % 2 == 0 else "NET NEGATİF",
            "notified": False,
        } for i in range(12)]
        result = measurement_summary(rows, now_ms=1_800_000_000_000)
        cohort = next(x for x in result["audiences"]["muted"] if x["kind"] == "scalp-forward")
        self.assertEqual((cohort["symbol"], cohort["family"]), ("SOLUSDT", "B3"))
        self.assertEqual(cohort["rolling"]["10"]["count"], 10)
        self.assertEqual(cohort["rolling"]["50"]["count"], 12)
        self.assertEqual(cohort["rolling"]["10"]["rate"], 0.5)
        self.assertEqual(len(cohort["rolling"]["10"]["wilson95"]), 2)


if __name__ == "__main__":
    unittest.main()
