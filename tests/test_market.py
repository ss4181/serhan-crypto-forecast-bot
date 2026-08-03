from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from crypto_forecaster.config import (
    DEFAULT_SYMBOLS,
    _configured_symbols,
    cache_path,
    market,
)
from crypto_forecaster.data import MARKET_ENDPOINTS, BinanceMarketDataClient
from crypto_forecaster.model import familywise_z


DATA = Path("data")


class MarketSelectionTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_the_traded_instrument_is_the_default(self) -> None:
        # The owner trades perpetuals; modelling spot would model a different
        # instrument that merely tracks it.
        self.assertEqual(market(), "futures")
        self.assertIn("fapi.binance.com", MARKET_ENDPOINTS[market()][0])

    @patch.dict(os.environ, {"CRYPTO_MARKET": "spot"}, clear=True)
    def test_spot_is_still_reachable(self) -> None:
        self.assertEqual(market(), "spot")
        self.assertEqual(BinanceMarketDataClient().page_limit, 1000)

    @patch.dict(os.environ, {"CRYPTO_MARKET": "options"}, clear=True)
    def test_an_unknown_market_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            market()

    def test_a_cache_cannot_be_mistaken_for_the_other_market(self) -> None:
        # Spot and futures prices differ; silently mixing them would corrupt a
        # model without any error ever being raised.
        with patch.dict(os.environ, {"CRYPTO_MARKET": "futures"}, clear=True):
            futures = cache_path(DATA, "BTCUSDT", "1h")
        with patch.dict(os.environ, {"CRYPTO_MARKET": "spot"}, clear=True):
            spot = cache_path(DATA, "BTCUSDT", "1h")
        self.assertNotEqual(futures, spot)
        self.assertIn("futures", futures.name)
        self.assertIn("spot", spot.name)


class SymbolSelectionTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_default_universe_is_unchanged(self) -> None:
        self.assertEqual(_configured_symbols(), DEFAULT_SYMBOLS)

    @patch.dict(os.environ, {"CRYPTO_SYMBOLS": "solusdt, 1000PEPEUSDT ,SOLUSDT"}, clear=True)
    def test_the_list_is_normalised_and_deduplicated(self) -> None:
        self.assertEqual(_configured_symbols(), ("SOLUSDT", "1000PEPEUSDT"))

    @patch.dict(os.environ, {"CRYPTO_SYMBOLS": "BTC/USDT"}, clear=True)
    def test_a_malformed_symbol_is_refused(self) -> None:
        # The symbol becomes a path segment, so its shape is not cosmetic.
        with self.assertRaises(ValueError):
            _configured_symbols()

    @patch.dict(os.environ, {"CRYPTO_SYMBOLS": "  "}, clear=True)
    def test_an_empty_list_falls_back(self) -> None:
        self.assertEqual(_configured_symbols(), DEFAULT_SYMBOLS)


class FamilywiseTests(unittest.TestCase):
    def test_the_correction_tightens_as_models_are_added(self) -> None:
        # Hardcoding the six-model value meant every added symbol quietly
        # weakened the correction, which is the direction that invents edges.
        self.assertAlmostEqual(familywise_z(6), 2.638257273476751, places=9)
        self.assertGreater(familywise_z(40), familywise_z(6))
        self.assertGreater(familywise_z(6), familywise_z(1))

    def test_a_single_model_still_gets_the_plain_interval(self) -> None:
        self.assertAlmostEqual(familywise_z(1), 1.959963984540054, places=9)


if __name__ == "__main__":
    unittest.main()
