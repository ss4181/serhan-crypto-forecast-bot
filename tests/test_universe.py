from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from crypto_forecaster.universe import load_trade1_universe


class Trade1UniverseTests(unittest.TestCase):
    def test_manifest_is_exactly_the_validated_30_plus_59_universe(self) -> None:
        manifest = load_trade1_universe()
        core = [entry for entry in manifest.entries if entry.group == "core30"]
        extended = [entry for entry in manifest.entries if entry.group == "extended59"]
        self.assertEqual(len(manifest.entries), 89)
        self.assertEqual(len(core), 30)
        self.assertEqual(len(extended), 59)
        self.assertEqual(manifest.scalp_horizons_minutes, (15, 30, 60))
        self.assertEqual(manifest.scalp_status, "research_only")

    def test_strategy_authority_does_not_leak_into_the_extended_group(self) -> None:
        manifest = load_trade1_universe()
        core = next(entry for entry in manifest.entries if entry.spot_symbol == "BTCUSDT")
        extended = next(entry for entry in manifest.entries if entry.spot_symbol == "RIFUSDT")
        self.assertEqual(core.trade1_strategies, ("S1", "S1+S4", "S2", "S3"))
        self.assertEqual(extended.trade1_strategies, ("S1", "S1+S4"))

    def test_spot_to_perpetual_contract_mapping_is_explicit(self) -> None:
        manifest = load_trade1_universe()
        expected = {
            "PEPEUSDT": "1000PEPEUSDT",
            "SHIBUSDT": "1000SHIBUSDT",
            "BONKUSDT": "1000BONKUSDT",
            "XECUSDT": "1000XECUSDT",
            "LUNCUSDT": "1000LUNCUSDT",
            "FLOKIUSDT": "1000FLOKIUSDT",
        }
        actual = {entry.spot_symbol: entry.perpetual_symbol for entry in manifest.entries}
        for spot, perpetual in expected.items():
            self.assertEqual(actual[spot], perpetual)

    @patch.dict(os.environ, {"CRYPTO_SCALP_SYMBOLS": "ethusdt, BTCUSDT,ethusdt"}, clear=True)
    def test_an_opt_in_subset_is_normalised_and_deduplicated(self) -> None:
        selected = load_trade1_universe().selected_entries()
        self.assertEqual([entry.spot_symbol for entry in selected], ["ETHUSDT", "BTCUSDT"])

    def test_a_symbol_outside_the_frozen_universe_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            load_trade1_universe().selected_entries("TRUMPUSDT")


if __name__ == "__main__":
    unittest.main()
