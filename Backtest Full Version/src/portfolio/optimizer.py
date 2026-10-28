"""
Portfolio Construction & Risk Parity Engine.
Implements Equal Weight, Minimum Variance, and Risk Parity (Equal Risk Contribution / ERC)
allocations, handles budget and no-shorting constraints, and provides strategy comparison.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize

from ..analytics.risk import calculate_risk_contributions, calculate_max_drawdown, calculate_sharpe_ratio
from ..analytics.returns import calculate_annualized_return, calculate_annualized_volatility


class PortfolioStrategy(str, Enum):
    FIXED = "fixed"
    EQUAL_WEIGHT = "equal_weight"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"


class PortfolioOptimizer:
    """
    Quantitative portfolio allocator enforcing budget and risk constraints.
    """

    def __init__(self, cov_matrix: pd.DataFrame, expected_returns: Optional[pd.Series] = None):
        self.cov_df = cov_matrix.copy()
        self.assets = list(cov_matrix.columns)
        self.n_assets = len(self.assets)
        self.expected_returns = expected_returns

        # Ensure positive-definiteness
        cov_vals = cov_matrix.values
        eigenvals, eigenvecs = np.linalg.eigh(cov_vals)
        clipped_evals = np.maximum(eigenvals, 1e-8)
        self.cov = eigenvecs @ np.diag(clipped_evals) @ eigenvecs.T

    def optimize_equal_weight(self) -> Dict[str, float]:
        """Equal-weight (1/N) allocation."""
        w = 1.0 / self.n_assets
        return {a: w for a in self.assets}

    def optimize_min_variance(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Minimum Variance Portfolio:
        min w^T * Sigma * w
        s.t. sum(w) = 1, min_w <= w_i <= max_w
        """
        cov = self.cov
        n = self.n_assets

        def port_var(w: np.ndarray) -> float:
            return float(w.T @ cov @ w)

        x0 = np.full(n, 1.0 / n)
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = optimize.minimize(
            port_var,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 500}
        )

        if not res.success:
            w_opt = x0
        else:
            w_opt = res.x / np.sum(res.x)

        return dict(zip(self.assets, w_opt.tolist()))

    def optimize_risk_parity(
        self,
        target_risk_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Equal Risk Contribution (ERC) / Risk Parity Portfolio:
        Finds w such that w_i * (Sigma * w)_i = b_i * (w^T * Sigma * w)
        For ERC, b_i = 1/N.
        Uses the convex objective: min 0.5 * w^T * Sigma * w - sum(b_i * ln(w_i))
        """
        cov = self.cov
        n = self.n_assets

        if target_risk_weights:
            b = np.array([target_risk_weights.get(a, 1.0 / n) for a in self.assets], dtype=float)
            b = b / np.sum(b)
        else:
            b = np.full(n, 1.0 / n)

        # Spinu (2013) convex formulation for Risk Parity
        def erc_objective(y: np.ndarray) -> float:
            return 0.5 * (y.T @ cov @ y) - np.sum(b * np.log(np.maximum(y, 1e-12)))

        def erc_gradient(y: np.ndarray) -> np.ndarray:
            return (cov @ y) - (b / np.maximum(y, 1e-12))

        y0 = np.full(n, 1.0 / np.sqrt(n))
        bounds = [(1e-6, None) for _ in range(n)]

        res = optimize.minimize(
            erc_objective,
            x0=y0,
            jac=erc_gradient,
            bounds=bounds,
            method="L-BFGS-B",
            options={"ftol": 1e-11, "maxiter": 500}
        )

        if res.success and np.sum(res.x) > 0:
            w_opt = res.x / np.sum(res.x)
        else:
            # Fallback to least-squares risk disparity
            def risk_budget_disparity(w: np.ndarray) -> float:
                sigma_p = np.sqrt(w.T @ cov @ w)
                mrc = (cov @ w) / max(1e-8, sigma_p)
                rc = w * mrc
                diff = rc - b * sigma_p
                return float(np.sum(diff ** 2))

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            res2 = optimize.minimize(
                risk_budget_disparity,
                x0=np.full(n, 1.0 / n),
                bounds=[(0.01, 1.0) for _ in range(n)],
                constraints=constraints,
                method="SLSQP"
            )
            w_opt = res2.x / np.sum(res2.x)

        return dict(zip(self.assets, w_opt.tolist()))

    def compare_strategies(
        self,
        asset_returns: pd.DataFrame,
        fixed_weights: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.04,
        periods_per_year: float = 252.0
    ) -> pd.DataFrame:
        """
        Produce side-by-side quantitative comparison across Fixed-Weight, Equal-Weight,
        Minimum Variance, and Risk Parity allocations on the same return history.
        """
        weights_dict: Dict[str, Dict[str, float]] = {}

        if fixed_weights:
            weights_dict["Fixed-Weight"] = fixed_weights

        weights_dict["Equal-Weight (1/N)"] = self.optimize_equal_weight()
        weights_dict["Minimum Variance"] = self.optimize_min_variance()
        weights_dict["Risk Parity (ERC)"] = self.optimize_risk_parity()

        rows = []
        for strat_name, w_map in weights_dict.items():
            w_arr = np.array([w_map.get(a, 0.0) for a in self.assets])
            strat_returns = asset_returns[self.assets].dot(w_arr)

            ann_ret = calculate_annualized_return(strat_returns, periods_per_year, geometric=True) * 100.0
            ann_vol = calculate_annualized_volatility(strat_returns, periods_per_year) * 100.0
            sharpe = calculate_sharpe_ratio(strat_returns, risk_free_rate, periods_per_year, geometric=True)
            mdd = calculate_max_drawdown(strat_returns, is_returns=True) * 100.0

            # Compute risk concentration: Herfindahl index of % risk contributions
            rc_df = calculate_risk_contributions(w_map, self.cov_df)
            prc = rc_df["Percentage Risk Contribution (%)"].values / 100.0
            herfindahl_rc = float(np.sum(prc ** 2))  # Lower = better diversified risk (1/N is optimal)

            # Weight breakdown string
            w_str = ", ".join(f"{a}: {w_map.get(a, 0.0)*100:.1f}%" for a in self.assets)

            rows.append({
                "Strategy": strat_name,
                "Ann. Return (%)": ann_ret,
                "Ann. Volatility (%)": ann_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": mdd,
                "Risk Herfindahl": herfindahl_rc,
                "Asset Weights": w_str
            })

        df_comp = pd.DataFrame(rows).set_index("Strategy")
        return df_comp
