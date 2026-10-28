"""
Unit tests for Correlation, Covariance, Rolling Correlation, Rolling Volatility,
and Dynamic Regime-Switching Correlation.
"""

import unittest
import numpy as np
import pandas as pd
from src.analytics.correlation import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_rolling_correlation,
    calculate_rolling_volatility,
    calculate_regime_switching_correlation,
    format_correlation_heatmap_text,
    RegimeCorrelationResult,
)


class TestCorrelationAnalytics(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=250, freq="B")
        r1 = np.random.normal(0, 0.01, 250)
        r2 = 0.8 * r1 + 0.2 * np.random.normal(0, 0.01, 250)
        r3 = -0.7 * r1 + 0.3 * np.random.normal(0, 0.01, 250)

        # Inject volatility cluster / stress shock into periods 150:200
        r1[150:200] *= 3.5
        r2[150:200] *= 3.5
        r3[150:200] *= 3.5

        self.returns_df = pd.DataFrame({
            "STOCK": r1,
            "ETF": r2,
            "INVERSE": r3
        }, index=dates)

    def test_correlation_matrix_properties(self):
        corr = calculate_correlation_matrix(self.returns_df)
        self.assertEqual(corr.shape, (3, 3))
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-6)
        self.assertGreater(corr.loc["STOCK", "ETF"], 0.7)
        self.assertLess(corr.loc["STOCK", "INVERSE"], -0.5)

    def test_covariance_matrix(self):
        cov = calculate_covariance_matrix(self.returns_df, periods_per_year=252)
        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue((np.diag(cov) > 0).all())

    def test_rolling_correlation(self):
        roll_corr = calculate_rolling_correlation(
            self.returns_df["STOCK"],
            self.returns_df["ETF"],
            window=30
        )
        self.assertGreater(len(roll_corr), 150)
        self.assertTrue((roll_corr >= -1.0).all() and (roll_corr <= 1.0).all())

    def test_rolling_volatility(self):
        roll_vol = calculate_rolling_volatility(self.returns_df, window=30, periods_per_year=252)
        self.assertEqual(roll_vol.shape, self.returns_df.shape)
        clean_vol = roll_vol.dropna()
        self.assertTrue((clean_vol > 0).all().all())

    def test_format_correlation_heatmap_text(self):
        corr = calculate_correlation_matrix(self.returns_df)
        text = format_correlation_heatmap_text(corr)
        self.assertIn("CORRELATION MATRIX", text)
        self.assertIn("STOCK", text)

    def test_regime_switching_correlation_volatility(self):
        result = calculate_regime_switching_correlation(
            self.returns_df,
            method="volatility_threshold",
            threshold=75.0,
            rolling_window=20
        )
        self.assertIsInstance(result, RegimeCorrelationResult)
        self.assertEqual(result.normal_corr.shape, (3, 3))
        self.assertEqual(result.crisis_corr.shape, (3, 3))
        self.assertEqual(result.correlation_shift.shape, (3, 3))
        self.assertGreater(result.regime_counts["Normal"], 50)
        self.assertGreater(result.regime_counts["Crisis"], 10)
        np.testing.assert_allclose(np.diag(result.correlation_shift), 0.0, atol=1e-6)

    def test_regime_switching_correlation_drawdown(self):
        result = calculate_regime_switching_correlation(
            self.returns_df,
            method="market_drawdown",
            threshold=5.0
        )
        self.assertIsInstance(result, RegimeCorrelationResult)
        self.assertEqual(result.normal_corr.shape, (3, 3))
        self.assertIn("Normal", result.regime_counts)

    def test_regime_correlation_summary_text(self):
        result = calculate_regime_switching_correlation(self.returns_df)
        summary = result.summary_text()
        self.assertIn("REGIME-SWITCHING CORRELATION", summary)
        self.assertIn("NORMAL / CALM REGIME", summary)
        self.assertIn("CRISIS / STRESS REGIME", summary)
        self.assertIn("CORRELATION SHIFT MATRIX", summary)


if __name__ == "__main__":
    unittest.main()
