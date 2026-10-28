"""
Unit tests for Walk-Forward Out-Of-Sample Validation and Macro Stress Testing.
"""

import unittest
import numpy as np
import pandas as pd

from src.analytics.validation import WalkForwardValidator, StressTestEngine


class TestValidationAndStress(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 5 years of daily returns (~1260 days)
        dates = pd.date_range("2018-01-01", periods=1260, freq="B")
        self.assets = ["CSPX", "BTC", "BANK", "GOLD"]

        # Mean and cov
        cov = np.diag([0.0256, 0.2500, 0.0400, 0.0196]) / 252.0
        rets = np.random.multivariate_normal(
            mean=[0.0004, 0.0008, 0.0005, 0.0002],
            cov=cov,
            size=1260
        )
        self.returns_df = pd.DataFrame(rets, index=dates, columns=self.assets)
        self.weights = {"CSPX": 0.45, "BTC": 0.25, "BANK": 0.25, "GOLD": 0.05}

    def test_walk_forward_execution(self):
        validator = WalkForwardValidator(self.returns_df, periods_per_year=252.0)
        df_oos = validator.run_walk_forward(
            strategy="fixed",
            fixed_weights=self.weights,
            train_days=504,  # 2 years train
            test_days=252,   # 1 year test
            step_days=252    # Step 1 year
        )
        self.assertGreaterEqual(len(df_oos), 2)
        self.assertIn("OOS 1", df_oos.index)
        self.assertIn("OOS Sharpe Ratio", df_oos.columns)
        self.assertIn("OOS Max Drawdown (%)", df_oos.columns)

    def test_walk_forward_insufficient_data_error(self):
        validator = WalkForwardValidator(self.returns_df.iloc[:200], periods_per_year=252.0)
        with self.assertRaises(ValueError):
            validator.run_walk_forward(train_days=500, test_days=100)

    def test_stress_test_equity_crash(self):
        df_stress = StressTestEngine.apply_scenario(
            weights=self.weights,
            initial_wealth=100_000.0,
            scenario_name="EQUITY_CRASH_2008"
        )
        self.assertIn("PORTFOLIO_TOTAL", df_stress.index)
        # Check that post-shock wealth is less than initial wealth
        pre_val = 100_000.0
        post_str = df_stress.loc["PORTFOLIO_TOTAL", "Post-Shock Value"].replace("$", "").replace(",", "")
        post_val = float(post_str)
        self.assertLess(post_val, pre_val)

        # Asset shock for CSPX must be -35.0%
        cspx_shock = df_stress.loc["CSPX", "Asset Shock (%)"]
        self.assertAlmostEqual(cspx_shock, -35.0, places=1)

    def test_stress_test_custom_scenario(self):
        custom = {"CSPX": -0.10, "BTC": +0.20, "BANK": -0.05, "GOLD": 0.0}
        df_custom = StressTestEngine.apply_scenario(
            weights=self.weights,
            initial_wealth=50_000.0,
            custom_shocks=custom
        )
        self.assertEqual(len(df_custom), 5)  # 4 assets + PORTFOLIO_TOTAL


if __name__ == "__main__":
    unittest.main()
