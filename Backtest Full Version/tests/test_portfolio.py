"""
Integration tests for the Portfolio orchestrator with synthetic data and baskets.
"""

import unittest
import pandas as pd
from src.portfolio.allocation import PortfolioConfig
from src.portfolio.portfolio import Portfolio
from src.data.provider import MockMarketDataProvider


class TestPortfolioIntegration(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
            "portfolio_name": "Default Balanced Wealth Portfolio",
            "base_currency": "IDR",
            "assets": {
                "CSPX": {
                    "type": "equity_etf",
                    "allocation": 0.45,
                    "ticker": "CSPX.L",
                    "description": "S&P 500 UCITS ETF"
                },
                "BTC": {
                    "type": "crypto",
                    "allocation": 0.25,
                    "ticker": "BTC-USD",
                    "description": "Bitcoin"
                },
                "BANK": {
                    "type": "basket",
                    "allocation": 0.25,
                    "description": "Indonesian Big Banks",
                    "constituents": {
                        "BBCA.JK": 0.40,
                        "BBRI.JK": 0.30,
                        "BMRI.JK": 0.30
                    }
                },
                "GOLD": {
                    "type": "commodity_or_gold_proxy",
                    "allocation": 0.05,
                    "ticker": "GLD",
                    "description": "Gold"
                }
            },
            "market_data": {
                "frequency": "daily",
                "start_date": "2021-01-01",
                "end_date": "2023-12-31",
                "calendar_alignment": "forward_fill",
                "outlier_std_threshold": 4.0
            }
        }
        self.config = PortfolioConfig.from_dict(self.config_dict)
        self.portfolio = Portfolio(self.config)
        self.provider = MockMarketDataProvider(seed=42)

    def test_allocation_breakdown_dataframe(self):
        df_alloc = self.portfolio.get_allocation_breakdown()
        self.assertEqual(len(df_alloc), 4)
        self.assertEqual(df_alloc["Portfolio Weight (%)"].sum(), 100.0)
        self.assertIn("Basket", df_alloc.loc[df_alloc["Asset"] == "BANK", "Ticker / Constituents"].values[0])

    def test_load_data_and_returns(self):
        report = self.portfolio.load_data(self.provider)
        self.assertIsNotNone(report)
        self.assertTrue(report.is_valid)

        self.assertIsNotNone(self.portfolio.asset_prices)
        self.assertIsNotNone(self.portfolio.asset_returns)
        self.assertIsNotNone(self.portfolio.portfolio_returns)

        self.assertEqual(list(self.portfolio.asset_prices.columns), ["CSPX", "BTC", "BANK", "GOLD"])
        self.assertEqual(list(self.portfolio.asset_returns.columns), ["CSPX", "BTC", "BANK", "GOLD"])

        # Check basket math
        aligned_df = self.portfolio.aligned_ticker_prices
        ret_bbca = aligned_df["BBCA.JK"].pct_change().dropna()
        ret_bbri = aligned_df["BBRI.JK"].pct_change().dropna()
        ret_bmri = aligned_df["BMRI.JK"].pct_change().dropna()
        expected_bank_ret = 0.40 * ret_bbca + 0.30 * ret_bbri + 0.30 * ret_bmri

        actual_bank_ret = self.portfolio.asset_returns["BANK"].dropna()
        pd.testing.assert_series_equal(actual_bank_ret, expected_bank_ret, check_names=False)

        # Check portfolio return calculation
        expected_port_ret = (
            0.45 * self.portfolio.asset_returns["CSPX"] +
            0.25 * self.portfolio.asset_returns["BTC"] +
            0.25 * self.portfolio.asset_returns["BANK"] +
            0.05 * self.portfolio.asset_returns["GOLD"]
        )
        pd.testing.assert_series_equal(self.portfolio.portfolio_returns, expected_port_ret, check_names=False)

    def test_performance_summary_table(self):
        self.portfolio.load_data(self.provider)
        summary = self.portfolio.get_performance_summary()
        self.assertIn("Portfolio", summary.index)
        self.assertIn("CSPX", summary.index)
        self.assertIn("BTC", summary.index)
        self.assertIn("BANK", summary.index)
        self.assertIn("GOLD", summary.index)
        self.assertIn("CAGR (%)", summary.columns)
        self.assertIn("Ann. Volatility (%)", summary.columns)


if __name__ == "__main__":
    unittest.main()