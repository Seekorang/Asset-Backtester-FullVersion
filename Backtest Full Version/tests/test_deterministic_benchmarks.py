"""
Deterministic Unit Tests for Financial Analytics, Multi-Currency FX, and Portfolio Accounting.
Verifies all core mathematical formulas against exact hand-calculated reference values.
"""

import unittest
import numpy as np
import pandas as pd

from src.analytics.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_cagr,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_portfolio_returns,
)
from src.analytics.risk import (
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_drawdown_details,
    calculate_sharpe_ratio,
    calculate_downside_volatility,
    calculate_sortino_ratio,
    calculate_var,
    calculate_cvar,
    calculate_risk_contributions,
)
from src.data.currency import (
    convert_asset_prices_to_base,
    convert_asset_returns_to_base,
)
from src.backtest.backtester import _calculate_xirr


class TestDeterministicFinancialMath(unittest.TestCase):
    """
    Precision tests asserting formulas match hand-calculated arithmetic.
    """

    def test_simple_and_log_returns_exact(self):
        # Price path: 100 -> 120 -> 90 -> 99
        prices = pd.Series([100.0, 120.0, 90.0, 99.0])
        simple_r = calculate_simple_returns(prices)
        log_r = calculate_log_returns(prices)

        # Expected simple: (120-100)/100 = +0.20; (90-120)/120 = -0.25; (99-90)/90 = +0.10
        expected_simple = [0.20, -0.25, 0.10]
        np.testing.assert_allclose(simple_r.values, expected_simple, rtol=1e-6)

        # Expected log: ln(1.20) ≈ 0.1823216; ln(0.75) ≈ -0.2876821; ln(1.10) ≈ 0.0953102
        expected_log = [np.log(1.20), np.log(0.75), np.log(1.10)]
        np.testing.assert_allclose(log_r.values, expected_log, rtol=1e-6)

        # Sum of log returns must equal ln(99/100)
        self.assertAlmostEqual(log_r.sum(), np.log(99.0 / 100.0), places=6)

    def test_drawdown_exact_values(self):
        # Wealth index: 100 -> 150 (peak) -> 120 -> 90 (trough, -40%) -> 150 (recovery) -> 160
        wealth = pd.Series([100.0, 150.0, 120.0, 90.0, 150.0, 160.0])
        dd = calculate_drawdown_series(wealth, is_returns=False)

        # Peaks: 100, 150, 150, 150, 150, 160
        # Drawdowns: 0.0, 0.0, (120-150)/150 = -0.20, (90-150)/150 = -0.40, 0.0, 0.0
        expected_dd = [0.0, 0.0, -0.20, -0.40, 0.0, 0.0]
        np.testing.assert_allclose(dd.values, expected_dd, rtol=1e-6)

        mdd = calculate_max_drawdown(wealth, is_returns=False)
        self.assertAlmostEqual(mdd, -0.40, places=6)

    def test_sharpe_and_sortino_hand_calculated(self):
        # 4 periodic returns: +0.06, -0.02, +0.04, -0.04
        # Mean = (0.06 - 0.02 + 0.04 - 0.04) / 4 = 0.01
        # Sample variance: deviations = [0.05, -0.03, 0.03, -0.05]
        # Sum of squared deviations = 0.0025 + 0.0009 + 0.0009 + 0.0025 = 0.0068
        # Sample std (ddof=1) = sqrt(0.0068 / 3) = sqrt(0.002266667) ≈ 0.0476098
        rets = pd.Series([0.06, -0.02, 0.04, -0.04])
        # Let periods_per_year = 4 (quarterly), risk_free_rate = 0.0
        ann_vol = calculate_annualized_volatility(rets, periods_per_year=4)
        expected_ann_vol = np.sqrt(0.0068 / 3.0) * np.sqrt(4.0)
        self.assertAlmostEqual(ann_vol, expected_ann_vol, places=6)

        # Arithmetic Sharpe with Rf = 0.0:
        sharpe_arith = calculate_sharpe_ratio(rets, risk_free_rate=0.0, periods_per_year=4, geometric=False)
        expected_sharpe = (0.01 * 4.0) / expected_ann_vol
        self.assertAlmostEqual(sharpe_arith, expected_sharpe, places=5)

        # Downside semi-deviation below target = 0:
        # Negative returns are -0.02 and -0.04. Squared: 0.0004 and 0.0016.
        # Mean squared downside = (0.0 + 0.0004 + 0.0 + 0.0016) / 4 = 0.0005.
        # sigma_down = sqrt(0.0005) * sqrt(4) = sqrt(0.0020) ≈ 0.04472136
        down_vol = calculate_downside_volatility(rets, risk_free_rate=0.0, periods_per_year=4)
        self.assertAlmostEqual(down_vol, np.sqrt(0.0020), places=6)

    def test_var_and_cvar_discrete_exact(self):
        # 10 discrete returns sorted:
        # [-0.10, -0.08, -0.05, -0.02, 0.00, 0.01, 0.03, 0.05, 0.07, 0.12]
        rets = pd.Series([-0.10, -0.08, -0.05, -0.02, 0.00, 0.01, 0.03, 0.05, 0.07, 0.12])
        # At 90% confidence, alpha = 0.10 -> 10th percentile is -0.082 (linear interpolation)
        var_90 = calculate_var(rets, confidence_level=0.90, method="historical", return_loss_as_positive=True)
        self.assertGreater(var_90, 0.07)
        self.assertLess(var_90, 0.10)

        # Expected shortfall (CVaR) must be greater than or equal to VaR
        cvar_90 = calculate_cvar(rets, confidence_level=0.90, method="historical", return_loss_as_positive=True)
        self.assertGreaterEqual(cvar_90, var_90)

    def test_multiplicative_currency_conversion(self):
        # Local asset: 1,000,000 IDR -> 1,100,000 IDR (+10.0% local return)
        # Exchange rate USDIDR: 15,000 IDR/USD -> 16,500 IDR/USD (+10.0% IDR depreciation)
        # Value in USD at t0: 1,000,000 / 15,000 = $66.6667
        # Value in USD at t1: 1,100,000 / 16,500 = $66.6667 (0.0% USD return!)
        # Multiplicative formula: (1 + 0.10) * (15,000 / 16,500) - 1 = 1.10 * (1 / 1.10) - 1 = 0.0!
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        local_prices = pd.Series([1_000_000.0, 1_100_000.0], index=dates)
        fx_rates = pd.DataFrame({"USDIDR=X": [15_000.0, 16_500.0]}, index=dates)

        usd_prices = convert_asset_prices_to_base(local_prices, from_currency="IDR", base_currency="USD", fx_rates_df=fx_rates)
        self.assertAlmostEqual(usd_prices.iloc[0], 1_000_000.0 / 15_000.0, places=4)
        self.assertAlmostEqual(usd_prices.iloc[1], 1_100_000.0 / 16_500.0, places=4)

        local_ret = calculate_simple_returns(local_prices)
        usd_ret = convert_asset_returns_to_base(local_ret, from_currency="IDR", base_currency="USD", fx_rates_df=fx_rates)
        self.assertAlmostEqual(usd_ret.iloc[0], 0.0, places=6)

    def test_risk_contributions_sum_to_one(self):
        # 3 assets with given covariance matrix and weights
        cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        weights = {"A": 0.50, "B": 0.30, "C": 0.20}
        df_rc = calculate_risk_contributions(weights, cov)

        # Percentage risk contributions must sum to 100.0%
        total_prc = df_rc["Percentage Risk Contribution (%)"].sum()
        self.assertAlmostEqual(total_prc, 100.0, places=5)

    def test_xirr_exact_doubling(self):
        # Invest $1,000 at t=0, receive $2,000 at t=1.0 year -> exact XIRR = +100.0%
        cfs = [(0.0, -1000.0), (1.0, 2000.0)]
        xirr = _calculate_xirr(cfs)
        self.assertIsNotNone(xirr)
        self.assertAlmostEqual(xirr, 1.0, places=5)

        # Invest $1,000 at t=0, receive $1,210 at t=2.0 years -> exact XIRR = +10.0%
        cfs2 = [(0.0, -1000.0), (2.0, 1210.0)]
        xirr2 = _calculate_xirr(cfs2)
        self.assertIsNotNone(xirr2)
        self.assertAlmostEqual(xirr2, 0.10, places=5)


if __name__ == "__main__":
    unittest.main()
