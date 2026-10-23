"""
Unit tests for Risk, Drawdown, Distribution Moments, and Tail Risk (VaR/CVaR) Analytics.
"""

import unittest
import numpy as np
import pandas as pd
from src.analytics.risk import (
    calculate_downside_volatility,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_skewness,
    calculate_kurtosis,
    calculate_var,
    calculate_cvar,
    calculate_annual_calendar_returns,
    compute_comprehensive_risk_report,
)


class TestRiskAnalytics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2021-01-01", "2023-12-31", freq="B")
        n = len(dates)

        r1 = np.random.normal(loc=0.0008, scale=0.01, size=n)
        r1[100:105] = -0.04  # Drawdown

        r2 = np.random.normal(loc=0.0005, scale=0.03, size=n)
        r2[200] = -0.15  # Shock

        self.returns_df = pd.DataFrame({
            "ASSET_A": r1,
            "ASSET_B": r2
        }, index=dates)

    def test_drawdown_series_and_max_drawdown(self):
        dd = calculate_drawdown_series(self.returns_df["ASSET_A"], is_returns=True)
        self.assertTrue((dd <= 1e-6).all())
        mdd = calculate_max_drawdown(self.returns_df["ASSET_A"], is_returns=True)
        self.assertLess(mdd, -0.15)
        self.assertGreater(mdd, -0.40)

    def test_sharpe_and_sortino_ratio(self):
        sharpe = calculate_sharpe_ratio(self.returns_df["ASSET_A"], risk_free_rate=0.02, periods_per_year=252)
        sortino = calculate_sortino_ratio(self.returns_df["ASSET_A"], risk_free_rate=0.02, periods_per_year=252)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(sortino, float)
        self.assertGreater(sortino, 0.0)

    def test_calmar_ratio(self):
        calmar = calculate_calmar_ratio(self.returns_df["ASSET_A"], is_returns=True, periods_per_year=252)
        self.assertIsInstance(calmar, float)
        self.assertGreater(calmar, 0.0)

    def test_skewness_and_kurtosis(self):
        skew = calculate_skewness(self.returns_df["ASSET_B"])
        kurt = calculate_kurtosis(self.returns_df["ASSET_B"], excess=True)
        self.assertLess(skew, 0.0)
        self.assertGreater(kurt, 0.0)

    def test_var_historical_and_parametric(self):
        var_hist_95 = calculate_var(self.returns_df["ASSET_A"], confidence_level=0.95, method="historical")
        var_param_95 = calculate_var(self.returns_df["ASSET_A"], confidence_level=0.95, method="parametric")
        var_cf_95 = calculate_var(self.returns_df["ASSET_A"], confidence_level=0.95, method="modified")

        self.assertGreater(var_hist_95, 0.0)
        self.assertGreater(var_param_95, 0.0)
        self.assertGreater(var_cf_95, 0.0)

        var_hist_99 = calculate_var(self.returns_df["ASSET_A"], confidence_level=0.99, method="historical")
        self.assertGreater(var_hist_99, var_hist_95)

    def test_cvar_expected_shortfall(self):
        var_95 = calculate_var(self.returns_df["ASSET_A"], confidence_level=0.95, method="historical")
        cvar_95 = calculate_cvar(self.returns_df["ASSET_A"], confidence_level=0.95, method="historical")
        self.assertGreaterEqual(cvar_95, var_95)

    def test_annual_calendar_returns_and_best_worst_year(self):
        ann_rets = calculate_annual_calendar_returns(self.returns_df)
        self.assertEqual(len(ann_rets), 3)

    def test_compute_comprehensive_risk_report(self):
        report = compute_comprehensive_risk_report(self.returns_df, risk_free_rate=0.02, periods_per_year=252)
        self.assertIn("ASSET_A", report.index)
        self.assertIn("ASSET_B", report.index)
        self.assertIn("Sharpe Ratio", report.columns)
        self.assertIn("Sortino Ratio", report.columns)
        self.assertIn("Max Drawdown (%)", report.columns)
        self.assertIn("Daily VaR 95% (%)", report.columns)
        self.assertIn("Daily CVaR 95% (%)", report.columns)


if __name__ == "__main__":
    unittest.main()