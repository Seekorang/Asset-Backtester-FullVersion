"""
Unit tests for Return Analytics & Performance Calculation.
"""

import unittest
import numpy as np
import pandas as pd
from src.analytics.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
    calculate_cagr,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_portfolio_returns,
    compute_return_summary
)


class TestReturnAnalytics(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        p1 = np.linspace(100.0, 200.0, len(dates))
        p2 = np.full(len(dates), 100.0)
        self.prices_df = pd.DataFrame({"GROWTH": p1, "FLAT": p2}, index=dates)

    def test_simple_returns(self):
        rets = calculate_simple_returns(self.prices_df)
        self.assertEqual(len(rets), len(self.prices_df) - 1)
        self.assertTrue((rets["GROWTH"] > 0).all())
        self.assertTrue((rets["FLAT"] == 0).all())

    def test_log_returns(self):
        log_r = calculate_log_returns(self.prices_df)
        self.assertEqual(len(log_r), len(self.prices_df) - 1)
        cum_log = log_r["GROWTH"].sum()
        expected_log = np.log(200.0 / 100.0)
        self.assertAlmostEqual(cum_log, expected_log, places=5)

    def test_cumulative_returns(self):
        rets = calculate_simple_returns(self.prices_df)
        cum_rets = calculate_cumulative_returns(rets)
        self.assertAlmostEqual(cum_rets["GROWTH"].iloc[-1], 1.0, places=5)
        self.assertAlmostEqual(cum_rets["FLAT"].iloc[-1], 0.0, places=5)

    def test_cagr_calculation(self):
        cagr = calculate_cagr(self.prices_df["GROWTH"])
        self.assertAlmostEqual(cagr, 1.0, delta=0.05)

    def test_annualized_metrics(self):
        rets = calculate_simple_returns(self.prices_df)
        ann_ret_geom = calculate_annualized_return(rets["GROWTH"], periods_per_year=252, geometric=True)
        ann_vol = calculate_annualized_volatility(rets["FLAT"], periods_per_year=252)

        self.assertAlmostEqual(ann_ret_geom, 1.0, delta=0.05)
        self.assertAlmostEqual(ann_vol, 0.0, places=5)

    def test_rebalanced_portfolio_returns(self):
        rets = calculate_simple_returns(self.prices_df)
        weights = {"GROWTH": 0.50, "FLAT": 0.50}
        port_ret = calculate_portfolio_returns(rets, weights, rebalanced=True)
        expected = 0.50 * rets["GROWTH"] + 0.50 * rets["FLAT"]
        pd.testing.assert_series_equal(port_ret, expected, check_names=False)

    def test_buy_and_hold_portfolio_returns(self):
        rets = calculate_simple_returns(self.prices_df)
        weights = {"GROWTH": 0.50, "FLAT": 0.50}
        port_ret_bh = calculate_portfolio_returns(rets, weights, rebalanced=False)
        cum_ret = (1.0 + port_ret_bh).prod() - 1.0
        self.assertAlmostEqual(cum_ret, 0.50, places=4)

    def test_compute_return_summary(self):
        summary = compute_return_summary(self.prices_df, is_returns=False)
        self.assertIn("GROWTH", summary.index)
        self.assertIn("FLAT", summary.index)
        self.assertIn("CAGR (%)", summary.columns)
        self.assertIn("Ann. Volatility (%)", summary.columns)
        self.assertAlmostEqual(summary.loc["FLAT", "Total Return (%)"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()