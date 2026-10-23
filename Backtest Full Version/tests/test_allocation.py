"""
Unit tests for Asset Allocation and Portfolio Configuration.
"""

import unittest
from src.portfolio.allocation import AssetConfig, PortfolioConfig


class TestAllocationConfig(unittest.TestCase):

    def setUp(self):
        self.valid_dict = {
            "portfolio_name": "Test Balanced Portfolio",
            "base_currency": "IDR",
            "assets": {
                "CSPX": {
                    "type": "equity_etf",
                    "allocation": 0.45,
                    "ticker": "CSPX.L",
                    "description": "S&P 500 ETF"
                },
                "BTC": {
                    "type": "crypto",
                    "allocation": 0.25,
                    "ticker": "BTC-USD"
                },
                "BANK": {
                    "type": "basket",
                    "allocation": 0.25,
                    "constituents": {
                        "BBCA.JK": 0.40,
                        "BBRI.JK": 0.30,
                        "BMRI.JK": 0.30
                    }
                },
                "GOLD": {
                    "type": "commodity_or_gold_proxy",
                    "allocation": 0.05,
                    "ticker": "GLD"
                }
            },
            "market_data": {
                "frequency": "daily",
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            }
        }

    def test_valid_portfolio_config(self):
        config = PortfolioConfig.from_dict(self.valid_dict)
        self.assertEqual(config.portfolio_name, "Test Balanced Portfolio")
        self.assertEqual(len(config.assets), 4)
        top_weights = config.get_top_level_allocations()
        self.assertAlmostEqual(sum(top_weights.values()), 1.0, places=4)
        self.assertEqual(top_weights["CSPX"], 0.45)
        self.assertEqual(top_weights["BTC"], 0.25)
        self.assertEqual(top_weights["BANK"], 0.25)
        self.assertEqual(top_weights["GOLD"], 0.05)

    def test_effective_ticker_weights(self):
        config = PortfolioConfig.from_dict(self.valid_dict)
        eff_weights = config.get_effective_ticker_weights()
        self.assertAlmostEqual(sum(eff_weights.values()), 1.0, places=4)
        self.assertAlmostEqual(eff_weights["CSPX.L"], 0.45)
        self.assertAlmostEqual(eff_weights["BTC-USD"], 0.25)
        self.assertAlmostEqual(eff_weights["BBCA.JK"], 0.25 * 0.40)  # 0.10
        self.assertAlmostEqual(eff_weights["BBRI.JK"], 0.25 * 0.30)  # 0.075
        self.assertAlmostEqual(eff_weights["BMRI.JK"], 0.25 * 0.30)  # 0.075
        self.assertAlmostEqual(eff_weights["GLD"], 0.05)

    def test_get_required_tickers(self):
        config = PortfolioConfig.from_dict(self.valid_dict)
        tickers = config.get_required_tickers()
        expected = sorted(["CSPX.L", "BTC-USD", "BBCA.JK", "BBRI.JK", "BMRI.JK", "GLD"])
        self.assertEqual(tickers, expected)

    def test_invalid_total_allocation_sum(self):
        invalid_dict = self.valid_dict.copy()
        invalid_dict["assets"] = {
            "CSPX": {"type": "equity_etf", "allocation": 0.50, "ticker": "CSPX.L"},
            "BTC": {"type": "crypto", "allocation": 0.30, "ticker": "BTC-USD"}
        }
        with self.assertRaises(ValueError) as ctx:
            PortfolioConfig.from_dict(invalid_dict)
        self.assertIn("violates the requirement of exactly 1.0", str(ctx.exception))

    def test_invalid_negative_allocation(self):
        invalid_dict = self.valid_dict.copy()
        invalid_dict["assets"] = {
            "CSPX": {"type": "equity_etf", "allocation": 1.20, "ticker": "CSPX.L"},
            "BTC": {"type": "crypto", "allocation": -0.20, "ticker": "BTC-USD"}
        }
        with self.assertRaises(ValueError):
            PortfolioConfig.from_dict(invalid_dict)

    def test_invalid_basket_constituents_sum(self):
        invalid_dict = self.valid_dict.copy()
        invalid_dict["assets"]["BANK"] = {
            "type": "basket",
            "allocation": 0.25,
            "constituents": {
                "BBCA.JK": 0.50,
                "BBRI.JK": 0.60  # Sum = 1.10
            }
        }
        with self.assertRaises(ValueError) as ctx:
            PortfolioConfig.from_dict(invalid_dict)
        self.assertIn("internal constituent weights sum to", str(ctx.exception))

    def test_single_asset_missing_ticker(self):
        invalid_dict = self.valid_dict.copy()
        invalid_dict["assets"]["CSPX"] = {
            "type": "equity_etf",
            "allocation": 0.45
        }
        with self.assertRaises(ValueError) as ctx:
            PortfolioConfig.from_dict(invalid_dict)
        self.assertIn("must define a 'ticker' symbol", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()