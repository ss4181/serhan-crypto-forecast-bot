from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from audit_research_registry import gate_maps


class ResearchRegistryTests(unittest.TestCase):
    def test_empty_never_passes(self):
        self.assertEqual(gate_maps({"gates": {}})[0]["claim"], "UNKNOWN_EMPTY_GATES")

    def test_missing_and_numeric_values_do_not_become_true_booleans(self):
        self.assertEqual(gate_maps({"gates": {"n": 100, "mean": 10}}), [])
        self.assertEqual(gate_maps({"gates": {"positive": None}}), [])
        self.assertEqual(gate_maps({"passed_all_gates": True}), [])

    def test_nested_variant_claims_are_kept_separate(self):
        rows = gate_maps({"variants": [{"gates": {"cost": True, "sample": False}},
                                       {"gates": {"cost": True, "sample": True}}]})
        self.assertEqual([r["claim"] for r in rows], ["RECORDED_CHECK_FAILED", "ALL_RECORDED_CHECKS_PASS"])
        self.assertNotEqual(rows[0]["path"], rows[1]["path"])


if __name__ == "__main__":
    unittest.main()
