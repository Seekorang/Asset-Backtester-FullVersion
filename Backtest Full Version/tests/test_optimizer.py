"""
Unit tests for Portfolio Optimizer and Risk Parity Engine.
"""

import unittest
import numpy as np
import pandas as pd

from src.portfolio.optimizer import PortfolioOptimizer, PortfolioStrategy
from src.analytics.risk import calculate_risk_contributions


class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        self.assets = ["CSPX", "BTC", "BANK", "GOLD"]
        # Realistic annualized covariance matrix
        self.cov_matrix = pd.DataFrame([
            [0.0256, 0.0050, 0.0030, 0.0010],
            [0.0050, 0.3600, 0.0040, 0.0020],
            [0.0030, 0.0040, 0.0400, 0.0015],
            [0.0010, 0.0020, 0.0015, 0.0196],
        ], index=self.assets, columns=self.assets)

        self.optimizer = PortfolioOptimizer(self.cov_matrix)

    def test_equal_weight_allocation(self):
        w_eq = self.optimizer.optimize_equal_weight()
        self.assertEqual(len(w_eq), 4)
        self.assertAlmostEqual(sum(w_eq.values()), 1.0, places=5)
        for a in self.assets:
            self.assertAlmostEqual(w_eq[a], 0.25, places=5)

    def test_minimum_variance_allocation(self):
        w_mv = self.optimizer.optimize_min_variance()
        self.assertAlmostEqual(sum(w_mv.values()), 1.0, places=5)
        # All weights must be within [0, 1]
        for a, weight in w_mv.items():
            self.assertGreaterEqual(weight, -1e-6)
            self.assertLessEqual(weight, 1.0 + 1e-6)

        # Min variance portfolio variance must be lower than equal weight variance
        w_eq_arr = np.array([0.25, 0.25, 0.25, 0.25])
        w_mv_arr = np.array([w_mv[a] for a in self.assets])

        var_eq = float(w_eq_arr.T @ self.cov_matrix.values @ w_eq_arr)
        var_mv = float(w_mv_arr.T @ self.cov_matrix.values @ w_mv_arr)
        self.assertLess(var_mv, var_eq)

    def test_risk_parity_equal_risk_contributions(self):
        w_rp = self.optimizer.optimize_risk_parity()
        self.assertAlmostEqual(sum(w_rp.values()), 1.0, places=5)

        # Compute risk contributions under risk parity weights
        rc_df = calculate_risk_contributions(w_rp, self.cov_matrix)
        prc = rc_df["Percentage Risk Contribution (%)"].values

        # In Equal Risk Contribution, each of the 4 assets should contribute ~25.0% of total risk
        for val in prc:
            self.assertAlmostEqual(val, 25.0, delta=1.5)

        # BTC (high vol 60%) must have significantly lower capital weight than Gold (low vol 14%)
        self.assertLess(w_rp["BTC"], w_rp["GOLD"])
        self.assertLess(w_rp["BTC"], w_rp["CSPX"])

    def test_strategy_comparison_table(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        rets = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.0004, 0.0008, 0.0005, 0.0002],
                cov=self.cov_matrix.values / 252.0,
                size=500
            ),
            index=dates,
            columns=self.assets
        )
        comp_df = self.optimizer.compare_strategies(
            asset_returns=rets,
            fixed_weights={"CSPX": 0.45, "BTC": 0.25, "BANK": 0.25, "GOLD": 0.05}
        )
        self.assertIn("Equal-Weight (1/N)", comp_df.index)
        self.assertIn("Minimum Variance", comp_df.index)
        self.assertIn("Risk Parity (ERC)", comp_df.index)
        self.assertIn("Fixed-Weight", comp_df.index)
        self.assertIn("Sharpe Ratio", comp_df.columns)
        self.assertIn("Risk Herfindahl", comp_df.columns)


if __name__ == "__main__":
    unittest.main()
