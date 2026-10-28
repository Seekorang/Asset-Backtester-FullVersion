"""
Unit tests for Benchmark Analytics Module:
Active return, weekly tracking error, weekly information ratio,
tracking error variance decomposition, comparative yield sensitivity,
and weekly return aggregation preserving full calendar compounding.
"""

import unittest
import numpy as np
import pandas as pd

from src.analytics.benchmark import (
    resample_returns_weekly,
    calculate_active_returns,
    calculate_excess_returns,
    calculate_tracking_error,
    calculate_tracking_error_decomposition,
    calculate_information_ratio,
    calculate_yield_sensitivity,
    compute_benchmark_report,
    BenchmarkReport,
)


class TestWeeklyReturns(unittest.TestCase):
    """Tests for resample_returns_weekly."""

    def setUp(self):
        dates = pd.date_range("2023-01-02", periods=60, freq="B")  # ~3 months
        np.random.seed(42)
        self.daily_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates, name="Portfolio")
        self.daily_df = pd.DataFrame({
            "A": np.random.normal(0.0003, 0.008, len(dates)),
            "B": np.random.normal(0.0001, 0.012, len(dates)),
        }, index=dates)

    def test_weekly_series_shape(self):
        weekly = resample_returns_weekly(self.daily_returns)
        self.assertGreater(len(weekly), 5)
        self.assertLess(len(weekly), len(self.daily_returns))

    def test_weekly_df_shape(self):
        weekly = resample_returns_weekly(self.daily_df)
        self.assertEqual(weekly.shape[1], 2)
        self.assertGreater(len(weekly), 5)

    def test_weekly_compounding(self):
        total_daily = float((1.0 + self.daily_returns).prod() - 1.0)
        weekly = resample_returns_weekly(self.daily_returns)
        total_weekly = float((1.0 + weekly).prod() - 1.0)
        self.assertAlmostEqual(total_daily, total_weekly, places=4)

    def test_weekend_returns_are_compounded(self):
        """Verify that weekend returns (Sat/Sun) are compounded into the Friday block, not dropped."""
        # 14 calendar days (Friday to Friday)
        dates = pd.date_range("2023-01-06", periods=15, freq="D")  # Fri to Fri
        rets = pd.Series(0.0, index=dates)
        # Put 10% gain on Saturday and 5% gain on Sunday
        sat_date = pd.Timestamp("2023-01-07")
        sun_date = pd.Timestamp("2023-01-08")
        if sat_date in rets.index:
            rets.loc[sat_date] = 0.10
        if sun_date in rets.index:
            rets.loc[sun_date] = 0.05

        weekly = resample_returns_weekly(rets, day="FRI")
        # Total compounded return of the series
        total_compounded = float((1.0 + rets).prod() - 1.0)
        total_weekly_compounded = float((1.0 + weekly).prod() - 1.0)
        self.assertAlmostEqual(total_compounded, total_weekly_compounded, places=6)
        # Verify that the ~15.5% weekend gain is present in the weekly return
        self.assertGreater(total_weekly_compounded, 0.15)

    def test_raises_on_non_datetime_index(self):
        bad = pd.Series([0.01, -0.005, 0.002], index=[0, 1, 2])
        with self.assertRaises(TypeError):
            resample_returns_weekly(bad)


class TestActiveReturns(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=500, freq="B")
        self.port = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates, name="Portfolio")
        self.bench = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates, name="Benchmark")

    def test_active_return_length(self):
        active = calculate_active_returns(self.port, self.bench)
        self.assertEqual(len(active), 500)

    def test_active_return_values(self):
        active = calculate_active_returns(self.port, self.bench)
        expected_first = self.port.iloc[0] - self.bench.iloc[0]
        self.assertAlmostEqual(active.iloc[0], expected_first, places=10)

    def test_excess_return_backward_compatibility_alias(self):
        active = calculate_active_returns(self.port, self.bench)
        excess = calculate_excess_returns(self.port, self.bench)
        pd.testing.assert_series_equal(active, excess, check_names=False)

    def test_partial_overlap(self):
        dates_a = pd.date_range("2020-01-02", periods=300, freq="B")
        dates_b = pd.date_range("2020-06-01", periods=300, freq="B")
        a = pd.Series(np.random.normal(0, 0.01, 300), index=dates_a)
        b = pd.Series(np.random.normal(0, 0.01, 300), index=dates_b)
        active = calculate_active_returns(a, b)
        self.assertLess(len(active), 300)
        self.assertGreater(len(active), 0)


class TestTrackingErrorAndDecomposition(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=500, freq="B")
        self.port = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
        self.bench = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)

    def test_tracking_error_positive(self):
        te = calculate_tracking_error(self.port, self.bench)
        self.assertGreater(te, 0.0)

    def test_tracking_error_zero_for_identical(self):
        te = calculate_tracking_error(self.port, self.port)
        self.assertAlmostEqual(te, 0.0, places=10)

    def test_tracking_error_decomposition_exact_match(self):
        """Var(R_p - R_b) = Var(R_p) + Var(R_b) - 2*Cov(R_p, R_b) must match direct formula."""
        decomp = calculate_tracking_error_decomposition(self.port, self.bench, periods_per_year=52.0)
        self.assertAlmostEqual(
            decomp["direct_tracking_error"],
            decomp["reconstructed_tracking_error"],
            places=6
        )
        self.assertGreater(decomp["direct_tracking_error"], 0.0)
        self.assertGreater(decomp["portfolio_volatility"], 0.0)
        self.assertGreater(decomp["benchmark_volatility"], 0.0)


class TestInformationRatio(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=500, freq="B")
        self.port = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
        self.bench = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)

    def test_ir_finite(self):
        ir = calculate_information_ratio(self.port, self.bench)
        self.assertTrue(np.isfinite(ir))

    def test_ir_zero_for_identical(self):
        ir = calculate_information_ratio(self.port, self.port)
        self.assertAlmostEqual(ir, 0.0, places=5)


class TestYieldSensitivity(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        dy = pd.Series(np.random.normal(0, 0.05, n), index=dates, name="dYield")
        self.port = pd.Series(-0.02 * dy.values + np.random.normal(0.0003, 0.01, n), index=dates)
        self.dy = dy

    def test_negative_beta(self):
        result = calculate_yield_sensitivity(self.port, self.dy)
        self.assertTrue(result["is_valid"])
        self.assertLess(result["beta"], 0.0)

    def test_r_squared_range(self):
        result = calculate_yield_sensitivity(self.port, self.dy)
        self.assertTrue(result["is_valid"])
        self.assertGreaterEqual(result["r_squared"], 0.0)
        self.assertLessEqual(result["r_squared"], 1.0)

    def test_significant_p_value(self):
        result = calculate_yield_sensitivity(self.port, self.dy)
        self.assertTrue(result["is_valid"])
        self.assertLess(result["p_value"], 0.05)

    def test_missing_data_marked_invalid(self):
        result = calculate_yield_sensitivity(self.port, None)
        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["beta"])
        self.assertIn("unavailable", result["error_message"])

    def test_insufficient_data_marked_invalid(self):
        short_port = self.port.iloc[:5]
        short_dy = self.dy.iloc[:5]
        result = calculate_yield_sensitivity(short_port, short_dy)
        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["beta"])
        self.assertIn("Insufficient", result["error_message"])


class TestBenchmarkReport(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=500, freq="B")
        self.port = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
        self.bench = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)
        dy = pd.Series(np.random.normal(0, 0.05, len(dates)), index=dates)
        self.dy = dy

    def test_report_without_yield(self):
        report = compute_benchmark_report(self.port, self.bench, yield_changes=None)
        self.assertIsInstance(report, BenchmarkReport)
        self.assertFalse(report.portfolio_yield_sensitivity["is_valid"])
        self.assertFalse(report.benchmark_yield_sensitivity["is_valid"])
        self.assertGreater(report.n_weekly_obs, 0)
        self.assertGreater(report.n_daily_obs, 0)

    def test_report_with_yield(self):
        report = compute_benchmark_report(self.port, self.bench, yield_changes=self.dy)
        self.assertIsInstance(report, BenchmarkReport)
        self.assertTrue(report.portfolio_yield_sensitivity["is_valid"])
        self.assertTrue(report.benchmark_yield_sensitivity["is_valid"])
        self.assertGreater(report.portfolio_yield_sensitivity["n_obs"], 0)
        self.assertGreater(report.benchmark_yield_sensitivity["n_obs"], 0)

    def test_summary_table_columns(self):
        report = compute_benchmark_report(self.port, self.bench, yield_changes=self.dy)
        df = report.summary_table()
        self.assertIn("Active Return (Annualized)", df.index)
        self.assertIn("Weekly Tracking Error", df.index)
        self.assertIn("Weekly Information Ratio", df.index)
        self.assertIn("Weekly Volatility (Ann.)", df.index)
        self.assertNotIn("Daily Volatility (Diluted)", df.index)
        self.assertNotIn("Daily Tracking Error", df.index)
        self.assertIn("10Y Treasury Beta", df.index)
        self.assertIn("Duration Risk Interpretation", df.index)

    def test_asset_attribution_table(self):
        dates = self.port.index
        assets_df = pd.DataFrame({
            "BTC": self.port * 2.0,
            "CSPX": self.bench,
        }, index=dates)
        weights = {"BTC": 0.5, "CSPX": 0.5}
        report = compute_benchmark_report(
            self.port, self.bench, yield_changes=self.dy,
            asset_returns=assets_df, weights=weights,
            irx_mean_bull=0.93, irx_mean_full=2.27
        )
        self.assertIsNotNone(report.asset_attribution_table)
        self.assertIn("BTC", report.asset_attribution_table.index)
        self.assertIn("Rebalancing / Compounding Effect", report.asset_attribution_table.index)
        self.assertIn("Total Active Return", report.asset_attribution_table.index)
        self.assertIn("Share of Active (%)", report.asset_attribution_table.columns)
        self.assertEqual(report.asset_attribution_table.loc["Total Active Return", "Share of Active (%)"], "100.0%")
        text = report.summary_text()
        self.assertIn("ATTRIBUTION NOTE (ACTIVE RETURN BREAKDOWN BY ASSET)", text)
        self.assertIn("actual 2016-2021 T-Bill yields [^IRX] averaged 0.93%", text)
        self.assertIn("Rebalancing/compounding effect", text)
        self.assertIn("frictionless daily rebalancing", text)
        self.assertIn("not a full Brinson attribution", text)
        self.assertIn("60/40 benchmark has no direct counterparts for BTC, Gold, or IDR Equities", text)

    def test_summary_text_comparative_yield_and_no_misleading_labels(self):
        report = compute_benchmark_report(self.port, self.bench, yield_changes=self.dy, has_bond_allocation=False)
        text = report.summary_text()
        # Verify strict terminology
        self.assertIn("BENCHMARK-RELATIVE PERFORMANCE & ATTRIBUTION AUDIT", text)
        self.assertIn("TRACKING ERROR DECOMPOSITION", text)
        self.assertIn("Active Return", text)
        # Verify NO misleading alpha labels or subjective IR ratings
        self.assertNotIn("Excess Return (Alpha)", text)
        self.assertNotIn("IR = GOOD", text)
        self.assertNotIn("EXCEPTIONAL active management", text)
        # Verify comparative yield sensitivity mentions both Portfolio and Benchmark
        self.assertIn("Portfolio Yield Beta", text)
        self.assertIn("Benchmark Yield Beta", text)
        # Verify typo fix (5 days/week)
        self.assertIn("5 days/week", text)
        self.assertNotIn("5 days/year", text)
        # Verify required wording
        self.assertIn("no direct duration exposure", text)

    def test_bnd_sanity_check_in_report(self):
        # Synthetic bond return negatively tied to yield changes (-0.05 * dy)
        bnd = -0.05 * self.dy + np.random.normal(0.0001, 0.002, len(self.dy))
        report = compute_benchmark_report(
            self.port, self.bench, yield_changes=self.dy, has_bond_allocation=False, bnd_returns=bnd
        )
        self.assertIsNotNone(report.bnd_yield_sensitivity)
        self.assertTrue(report.bnd_yield_sensitivity["is_valid"])
        self.assertLess(report.bnd_yield_sensitivity["beta"], 0.0)
        self.assertGreater(report.bnd_yield_sensitivity["implied_duration"], 0.0)
        text = report.summary_text()
        self.assertIn("BND Sanity Check", text)
        self.assertIn("no direct duration exposure", text)


if __name__ == "__main__":
    unittest.main()
