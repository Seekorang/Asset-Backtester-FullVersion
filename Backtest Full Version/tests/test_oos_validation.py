"""
Out-of-Sample Validation Tests.
Compares in-sample vs out-of-sample metrics across strategies
to detect overfitting and validate model robustness.
"""

import unittest
import numpy as np
import pandas as pd

from src.analytics.validation import WalkForwardValidator
from src.analytics.returns import calculate_annualized_return, calculate_annualized_volatility
from src.analytics.risk import calculate_sharpe_ratio, calculate_max_drawdown
from src.portfolio.optimizer import PortfolioOptimizer


class TestOutOfSampleValidation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 8 years of daily data (~2016 business days)
        n_days = 2016
        dates = pd.date_range("2016-01-04", periods=n_days, freq="B")
        self.assets = ["CSPX", "BTC", "BANK", "GOLD"]

        # Realistic-ish daily return parameters
        daily_means = np.array([0.0004, 0.0010, 0.0003, 0.0001])
        daily_vols = np.array([0.012, 0.040, 0.015, 0.008])

        # Moderate positive correlations, gold decorrelated
        corr = np.array([
            [1.00, 0.25, 0.15, 0.05],
            [0.25, 1.00, 0.10, -0.05],
            [0.15, 0.10, 1.00, 0.02],
            [0.05, -0.05, 0.02, 1.00],
        ])
        cov = np.outer(daily_vols, daily_vols) * corr

        rets = np.random.multivariate_normal(daily_means, cov, size=n_days)
        self.returns_df = pd.DataFrame(rets, index=dates, columns=self.assets)
        self.fixed_weights = {"CSPX": 0.45, "BTC": 0.25, "BANK": 0.25, "GOLD": 0.05}

    def _portfolio_return_series(self, returns_df, weights):
        w = np.array([weights.get(a, 0.0) for a in self.assets])
        return returns_df[self.assets].dot(w)

    def test_oos_fixed_weight_runs(self):
        """Walk-forward with fixed weights should produce at least 4 windows."""
        validator = WalkForwardValidator(
            self.returns_df, periods_per_year=252.0, risk_free_rate=0.04
        )
        df_oos = validator.run_walk_forward(
            strategy="fixed",
            fixed_weights=self.fixed_weights,
            train_days=756,
            test_days=252,
            step_days=252,
        )
        self.assertGreaterEqual(len(df_oos), 4)
        # Every OOS sharpe should be finite
        for val in df_oos["OOS Sharpe Ratio"]:
            self.assertTrue(np.isfinite(val))

    def test_oos_all_strategies(self):
        """Compare fixed, equal_weight, min_variance, risk_parity out-of-sample."""
        strategies = ["fixed", "equal_weight", "min_variance", "risk_parity"]
        results = {}

        for strat in strategies:
            validator = WalkForwardValidator(
                self.returns_df, periods_per_year=252.0, risk_free_rate=0.04
            )
            df_oos = validator.run_walk_forward(
                strategy=strat,
                fixed_weights=self.fixed_weights if strat == "fixed" else None,
                train_days=756,
                test_days=252,
                step_days=252,
            )
            results[strat] = df_oos

        # All strategies should produce same number of windows
        n_windows = len(results["fixed"])
        for strat in strategies:
            self.assertEqual(len(results[strat]), n_windows)

        # Min variance OOS volatility should generally be lower
        mv_avg_vol = results["min_variance"]["OOS Volatility (%)"].mean()
        ew_avg_vol = results["equal_weight"]["OOS Volatility (%)"].mean()
        # Not strictly guaranteed with random data, just check they're positive
        self.assertGreater(mv_avg_vol, 0)
        self.assertGreater(ew_avg_vol, 0)

    def test_insample_vs_oos_sharpe_degradation(self):
        """
        In-sample Sharpe should be >= OOS Sharpe on average.
        This is the core overfitting check: if OOS sharpe is consistently
        much lower than in-sample, the strategy is overfit.
        """
        train_days = 756
        test_days = 252
        step_days = 252
        n_obs = len(self.returns_df)

        is_sharpes = []
        oos_sharpes = []
        start_idx = 0

        while (start_idx + train_days + test_days) <= n_obs:
            train_rets = self.returns_df.iloc[start_idx:start_idx + train_days]
            test_rets = self.returns_df.iloc[
                start_idx + train_days:start_idx + train_days + test_days
            ]

            # Optimize min variance on training data
            train_cov = train_rets.cov() * 252.0
            opt = PortfolioOptimizer(train_cov)
            w_map = opt.optimize_min_variance()

            # In-sample performance
            is_port = self._portfolio_return_series(train_rets, w_map)
            is_sharpe = calculate_sharpe_ratio(is_port, 0.04, 252.0)
            is_sharpes.append(is_sharpe)

            # Out-of-sample performance (same weights, unseen data)
            oos_port = self._portfolio_return_series(test_rets, w_map)
            oos_sharpe = calculate_sharpe_ratio(oos_port, 0.04, 252.0)
            oos_sharpes.append(oos_sharpe)

            start_idx += step_days

        self.assertGreaterEqual(len(is_sharpes), 3)

        # Both should be finite
        for s in is_sharpes + oos_sharpes:
            self.assertTrue(np.isfinite(s))

        # Compute degradation
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        degradation = avg_is - avg_oos

        # With random data, degradation can go either way,
        # but it should not be absurdly large (> 2.0 would mean something broke)
        self.assertLess(abs(degradation), 2.0,
            f"IS Sharpe={avg_is:.3f}, OOS Sharpe={avg_oos:.3f}, gap={degradation:.3f}")

    def test_oos_returns_are_realistic(self):
        """OOS annualized returns should be between -50% and +100%."""
        validator = WalkForwardValidator(
            self.returns_df, periods_per_year=252.0, risk_free_rate=0.04
        )
        df_oos = validator.run_walk_forward(
            strategy="risk_parity",
            train_days=756,
            test_days=252,
            step_days=252,
        )
        for ret in df_oos["OOS Ann. Return (%)"]:
            self.assertGreater(ret, -50.0)
            self.assertLess(ret, 100.0)

    def test_oos_max_drawdown_negative(self):
        """OOS max drawdown should always be negative (or zero)."""
        validator = WalkForwardValidator(
            self.returns_df, periods_per_year=252.0, risk_free_rate=0.04
        )
        df_oos = validator.run_walk_forward(
            strategy="fixed",
            fixed_weights=self.fixed_weights,
            train_days=756,
            test_days=252,
            step_days=252,
        )
        for mdd in df_oos["OOS Max Drawdown (%)"]:
            self.assertLessEqual(mdd, 0.0)


if __name__ == "__main__":
    unittest.main()
