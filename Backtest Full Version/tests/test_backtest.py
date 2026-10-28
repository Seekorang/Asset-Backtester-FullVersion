"""
Unit tests for Historical Backtesting Engine with Tax-Efficient Inflow Rebalancing.
"""

import unittest
import numpy as np
import pandas as pd

from src.portfolio.allocation import PortfolioConfig
from src.portfolio.portfolio import Portfolio
from src.data.provider import MockMarketDataProvider
from src.backtest.backtester import Backtester, BacktestConfig, BacktestResult, RebalanceFrequency


class TestBacktester(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
            "portfolio_name": "Test Backtest Portfolio",
            "base_currency": "IDR",
            "assets": {
                "CSPX": {"type": "equity_etf", "allocation": 0.45, "ticker": "CSPX.L"},
                "BTC": {"type": "crypto", "allocation": 0.25, "ticker": "BTC-USD"},
                "BANK": {"type": "equity", "allocation": 0.25, "ticker": "BBCA.JK"},
                "GOLD": {"type": "commodity_or_gold_proxy", "allocation": 0.05, "ticker": "GLD"}
            },
            "market_data": {
                "frequency": "daily",
                "start_date": "2020-01-01",
                "end_date": "2022-12-31",
                "calendar_alignment": "forward_fill"
            }
        }
        self.portfolio_config = PortfolioConfig.from_dict(self.config_dict)
        self.portfolio = Portfolio(self.portfolio_config)
        self.provider = MockMarketDataProvider(seed=42)
        self.portfolio.load_data(self.provider)

        self.backtester = Backtester()

    def test_basic_backtest_no_contribution(self):
        cfg = BacktestConfig(
            initial_capital=100_000_000.0,
            monthly_contribution=0.0,
            contribution_growth_rate=0.0,
            rebalance_frequency="quarterly",
            transaction_cost_pct=0.001,
            annual_inflation_rate=0.03
        )
        res = self.backtester.run(self.portfolio, cfg)
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.total_contributions, 0.0)
        self.assertEqual(res.total_capital_invested, 100_000_000.0)
        self.assertGreater(res.final_nominal_wealth, 0.0)
        self.assertGreater(res.rebalance_events_count, 5)
        self.assertAlmostEqual(res.final_nominal_wealth, res.total_capital_invested + res.total_investment_gains, places=2)

    def test_backtest_with_monthly_contributions(self):
        cfg = BacktestConfig(
            initial_capital=100_000_000.0,
            monthly_contribution=5_000_000.0,
            contribution_growth_rate=0.10,
            rebalance_frequency="quarterly",
            transaction_cost_pct=0.001,
            annual_inflation_rate=0.03
        )
        res = self.backtester.run(self.portfolio, cfg)
        self.assertGreater(res.total_contributions, 100_000_000.0)
        self.assertGreater(res.total_capital_invested, 200_000_000.0)
        self.assertGreater(res.final_nominal_wealth, res.final_real_wealth)

    def test_rebalance_frequencies(self):
        for freq in ["monthly", "quarterly", "semi_annual", "annual", "none", "threshold", "inflow_only"]:
            cfg = BacktestConfig(
                initial_capital=50_000_000.0,
                monthly_contribution=1_000_000.0,
                rebalance_frequency=freq,
                rebalance_threshold=0.05
            )
            res = self.backtester.run(self.portfolio, cfg)
            self.assertIsInstance(res, BacktestResult)
            if freq in ("none", "inflow_only"):
                self.assertEqual(res.rebalance_events_count, 0)
            elif freq == "annual":
                self.assertLessEqual(res.rebalance_events_count, 4)

    def test_inflow_rebalancing_steering(self):
        # Tax-efficient inflow steering with zero physical sales
        cfg = BacktestConfig(
            initial_capital=50_000_000.0,
            monthly_contribution=2_000_000.0,
            rebalance_frequency="inflow_only",
            inflow_rebalancing=True
        )
        res = self.backtester.run(self.portfolio, cfg)
        self.assertEqual(res.rebalance_events_count, 0)
        self.assertTrue(res.inflow_rebalancing_used)
        self.assertGreater(res.final_nominal_wealth, 50_000_000.0)

    def test_compare_rebalancing_strategies(self):
        df_comp = self.backtester.compare_rebalancing_strategies(self.portfolio)
        self.assertIn("Quarterly Calendar (Standard)", df_comp.index)
        self.assertIn("Inflow-Only (Zero Sales / Pure DCA Steering)", df_comp.index)
        self.assertIn("Ending Wealth", df_comp.columns)
        self.assertIn("Sales Executed", df_comp.columns)
        self.assertIn("Tax-Efficiency", df_comp.columns)

    def test_summary_table(self):
        cfg = BacktestConfig(initial_capital=100_000_000.0, monthly_contribution=5_000_000.0, inflow_rebalancing=True)
        res = self.backtester.run(self.portfolio, cfg)
        df_summary = res.summary_table()
        self.assertIn("Initial Capital", df_summary.index)
        self.assertIn("Ending Nominal Wealth", df_summary.index)
        self.assertIn("Ending Real Wealth (Today's Value)", df_summary.index)
        self.assertIn("Tax-Efficient Inflow Steering", df_summary.index)

    def test_avoided_turnover_exact_formula(self):
        cfg = BacktestConfig(
            initial_capital=50_000_000.0,
            monthly_contribution=1_000_000.0,
            rebalance_frequency="quarterly",
            tolerance_band=0.15,
            inflow_rebalancing=True
        )
        res = self.backtester.run(self.portfolio, cfg)
        self.assertGreater(res.estimated_turnover_avoided, 0.0)


if __name__ == "__main__":
    unittest.main()
