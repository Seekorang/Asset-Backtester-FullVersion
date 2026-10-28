"""
Research Validation Engine.
Provides Walk-Forward Out-Of-Sample (OOS) rolling evaluation and
Macro Stress-Testing under extreme historical and synthetic shock scenarios.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .risk import calculate_sharpe_ratio, calculate_max_drawdown
from .returns import calculate_annualized_return, calculate_annualized_volatility
from ..portfolio.optimizer import PortfolioOptimizer


class WalkForwardValidator:
    """
    Evaluates allocation strategies out-of-sample across rolling chronological windows.
    Prevents lookahead bias by estimating parameters strictly on in-sample training data.
    """

    def __init__(
        self,
        asset_returns: pd.DataFrame,
        periods_per_year: float = 252.0,
        risk_free_rate: float = 0.04
    ):
        self.returns = asset_returns.copy()
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.dates = self.returns.index
        self.assets = list(self.returns.columns)

    def run_walk_forward(
        self,
        strategy: str = "risk_parity",  # 'fixed', 'equal_weight', 'risk_parity', 'min_variance'
        fixed_weights: Optional[Dict[str, float]] = None,
        train_days: int = 756,  # ~3 years in-sample training
        test_days: int = 252,   # ~1 year out-of-sample testing
        step_days: int = 252    # Step window by 1 year
    ) -> pd.DataFrame:
        """
        Execute rolling chronological train/test walk-forward validation.
        """
        n_obs = len(self.dates)
        if n_obs < (train_days + test_days):
            raise ValueError(
                f"Total observations ({n_obs}) is less than train_days ({train_days}) + test_days ({test_days})."
            )

        records = []
        window_idx = 1
        start_idx = 0

        while (start_idx + train_days + test_days) <= n_obs:
            train_start = self.dates[start_idx]
            train_end = self.dates[start_idx + train_days - 1]
            test_start = self.dates[start_idx + train_days]
            test_end = self.dates[start_idx + train_days + test_days - 1]

            train_rets = self.returns.loc[train_start:train_end]
            test_rets = self.returns.loc[test_start:test_end]

            # In-sample calibration: determine weights strictly from train data
            if strategy == "fixed" and fixed_weights:
                w_map = fixed_weights
            elif strategy == "equal_weight":
                w_map = {a: 1.0 / len(self.assets) for a in self.assets}
            else:
                train_cov = train_rets.cov() * self.periods_per_year
                opt = PortfolioOptimizer(train_cov)
                if strategy == "min_variance":
                    w_map = opt.optimize_min_variance()
                elif strategy == "risk_parity":
                    w_map = opt.optimize_risk_parity()
                else:
                    w_map = {a: 1.0 / len(self.assets) for a in self.assets}

            # Out-of-sample execution: evaluate weights on future test data
            w_arr = np.array([w_map.get(a, 0.0) for a in self.assets])
            oos_portfolio_rets = test_rets[self.assets].dot(w_arr)

            oos_ann_ret = calculate_annualized_return(oos_portfolio_rets, self.periods_per_year, geometric=True) * 100.0
            oos_ann_vol = calculate_annualized_volatility(oos_portfolio_rets, self.periods_per_year) * 100.0
            oos_sharpe = calculate_sharpe_ratio(oos_portfolio_rets, self.risk_free_rate, self.periods_per_year, geometric=True)
            oos_mdd = calculate_max_drawdown(oos_portfolio_rets, is_returns=True) * 100.0

            records.append({
                "Window": f"OOS {window_idx}",
                "Train Period": f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
                "Test Period": f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                "Strategy": strategy,
                "OOS Ann. Return (%)": oos_ann_ret,
                "OOS Volatility (%)": oos_ann_vol,
                "OOS Sharpe Ratio": oos_sharpe,
                "OOS Max Drawdown (%)": oos_mdd,
            })

            start_idx += step_days
            window_idx += 1

        df_oos = pd.DataFrame(records).set_index("Window")
        return df_oos


class StressTestEngine:
    """
    Hypothetical & Historical Macro Stress Testing.
    Applies severe factor shocks to inspect tail risk, portfolio drawdown, and risk attribution.
    """

    PREDEFINED_SCENARIOS: Dict[str, Dict[str, float]] = {
        "EQUITY_CRASH_2008": {
            "CSPX": -0.35,  # S&P 500 plunges 35%
            "BTC": -0.45,   # Crypto sells off aggressively
            "BANK": -0.28,  # Banking sector contracts
            "GOLD": +0.12,  # Flight to quality into gold
        },
        "CRYPTO_WINTER": {
            "CSPX": -0.05,
            "BTC": -0.65,   # Bitcoin collapses 65%
            "BANK": -0.02,
            "GOLD": +0.04,
        },
        "EM_CURRENCY_CRISIS_USD_SPIKE": {
            "CSPX": -0.08,
            "BTC": -0.15,
            "BANK": -0.35,  # Indonesian banks hit hard in USD terms due to FX devaluation
            "GOLD": +0.15,
        },
        "STAGFLATION_INFLATION_SHOCK": {
            "CSPX": -0.22,
            "BTC": -0.25,
            "BANK": -0.15,
            "GOLD": +0.28,  # Gold acts as prime commodity hedge
        },
        "GLOBAL_LIQUIDITY_RUN": {
            "CSPX": -0.20,
            "BTC": -0.30,
            "BANK": -0.20,
            "GOLD": -0.10,  # Even gold sold for margin liquidity
        }
    }

    @classmethod
    def apply_scenario(
        cls,
        weights: Dict[str, float],
        initial_wealth: float = 100_000.0,
        scenario_name: str = "EQUITY_CRASH_2008",
        custom_shocks: Optional[Dict[str, float]] = None,
        currency_symbol: str = "$"
    ) -> pd.DataFrame:
        """
        Evaluate instantaneous portfolio wealth impact and asset loss contribution under a macro shock.
        """
        shocks = custom_shocks or cls.PREDEFINED_SCENARIOS.get(scenario_name, {})
        if not shocks:
            raise ValueError(f"Unknown scenario: '{scenario_name}'. Provide custom_shocks or choose a valid scenario.")

        total_w = sum(weights.values())
        norm_w = {a: w / total_w for a, w in weights.items()}

        rows = []
        portfolio_shock = 0.0

        for a, w in norm_w.items():
            shock_pct = shocks.get(a, 0.0)
            initial_asset_val = initial_wealth * w
            asset_loss = initial_asset_val * shock_pct
            post_shock_val = initial_asset_val + asset_loss
            portfolio_shock += w * shock_pct

            rows.append({
                "Asset": a,
                "Weight (%)": w * 100.0,
                "Asset Shock (%)": shock_pct * 100.0,
                "Pre-Shock Value": f"{currency_symbol}{initial_asset_val:,.2f}",
                "Dollar Impact": f"{currency_symbol}{asset_loss:+,.2f}",
                "Post-Shock Value": f"{currency_symbol}{post_shock_val:,.2f}",
            })

        post_wealth = initial_wealth * (1.0 + portfolio_shock)
        total_dollar_impact = post_wealth - initial_wealth

        df_summary = pd.DataFrame(rows).set_index("Asset")
        df_summary.loc["PORTFOLIO_TOTAL"] = [
            100.0,
            portfolio_shock * 100.0,
            f"{currency_symbol}{initial_wealth:,.2f}",
            f"{currency_symbol}{total_dollar_impact:+,.2f}",
            f"{currency_symbol}{post_wealth:,.2f}"
        ]
        return df_summary
