"""
Historical Backtesting Engine.
Simulates portfolio execution with initial capital, dynamic monthly contributions,
annual contribution escalation, calendar, threshold, and tax-efficient inflow rebalancing,
transaction costs, and inflation-adjusted real purchasing power tracking without look-ahead bias.
Disentangles pure investment performance (TWR / MWR) from wealth accumulation via deposits.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from scipy import optimize

from ..portfolio.portfolio import Portfolio
from ..analytics.returns import calculate_cagr, calculate_simple_returns
from ..analytics.risk import calculate_max_drawdown, calculate_drawdown_series


class RebalanceFrequency(str, Enum):
    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    THRESHOLD = "threshold"
    INFLOW_ONLY = "inflow_only"  # Tax-efficient: rebalance purely through incoming contributions; zero asset sales


@dataclass
class BacktestConfig:
    """
    Parameters configuring the historical backtesting simulation.
    """
    initial_capital: float = 10_000.0  # Initial portfolio capital
    monthly_contribution: float = 500.0  # Periodic monthly contribution
    contribution_growth_rate: float = 0.10  # Annual escalation rate of monthly contributions
    rebalance_frequency: str = "quarterly"  # monthly, quarterly, semi_annual, annual, threshold, inflow_only, none
    rebalance_threshold: float = 0.05  # Allocation deviation trigger for threshold rebalancing
    transaction_cost_pct: float = 0.001  # 0.10% (10 bps) fee & slippage per trade
    annual_inflation_rate: float = 0.03  # Annual inflation assumption
    currency_symbol: str = "$"  # Currency symbol ($ for USD, Rp for IDR)
    # Tax-Efficient Inflow Rebalancing Options:
    inflow_rebalancing: bool = False  # Direct monthly deposits exclusively to underweight assets to minimize selling
    tolerance_band: Optional[float] = None  # Wider tolerance corridor before triggering asset sales
    capital_gains_tax_rate: float = 0.0  # Capital gains tax rate on sales (e.g. 0.001 for ID PPh Final, 0.20 for foreign CGT)


@dataclass
class BacktestResult:
    """
    Comprehensive output of the historical backtest run with institutional cash-flow attribution.
    """
    history_df: pd.DataFrame
    config: BacktestConfig
    target_weights: Dict[str, float]
    initial_capital: float
    final_nominal_wealth: float
    final_real_wealth: float
    total_contributions: float
    total_capital_invested: float
    total_investment_gains: float
    total_transaction_costs: float
    # True Investment Performance (Independent of external deposits)
    twr_nominal: float  # Cumulative Time-Weighted Return
    twr_cagr_nominal: float  # Annualized Time-Weighted Return (Pure Investment CAGR)
    twr_real: float  # Real Cumulative Time-Weighted Return
    twr_cagr_real: float  # Real Annualized Time-Weighted Return
    mwr_xirr: Optional[float]  # Money-Weighted Return / XIRR
    # Total Wealth Growth (Includes contributions)
    cagr_nominal: float  # Ratio CAGR: (Ending / Initial)**(1/Y) - 1 (kept for backwards compatibility)
    cagr_real: float
    max_drawdown: float
    rebalance_events_count: int
    duration_years: float
    # Tax-Efficiency & Friction Metrics
    inflow_rebalancing_used: bool = False
    estimated_turnover_avoided: float = 0.0
    estimated_tax_savings: float = 0.0

    def summary_table(self) -> pd.DataFrame:
        """Format high-level key backtest metrics into a clean, institutional summary table."""
        sym = self.config.currency_symbol
        mwr_str = f"{self.mwr_xirr * 100:+.2f}%" if self.mwr_xirr is not None else "N/A"
        rebal_mode_str = "Active (Cash Flow Steering)" if self.inflow_rebalancing_used else "Standard (Fixed Weight)"

        summary_rows = [
            ("Initial Capital", f"{sym}{self.initial_capital:,.2f}"),
            ("Total Monthly Contributions", f"{sym}{self.total_contributions:,.2f}"),
            ("Total Capital Invested", f"{sym}{self.total_capital_invested:,.2f}"),
            ("Total Investment Gains (Net of Fees)", f"{sym}{self.total_investment_gains:,.2f}"),
            ("Ending Nominal Wealth", f"{sym}{self.final_nominal_wealth:,.2f}"),
            ("Ending Real Wealth (Today's Value)", f"{sym}{self.final_real_wealth:,.2f}"),
            ("Nominal Wealth Multiplier", f"{self.final_nominal_wealth / max(1.0, self.total_capital_invested):.2f}x"),
            ("Real Wealth Multiplier", f"{self.final_real_wealth / max(1.0, self.total_capital_invested):.2f}x"),
            ("Nominal Portfolio CAGR", f"{self.twr_cagr_nominal * 100:+.2f}%"),
            ("Real Portfolio CAGR (Inflation Adj)", f"{self.twr_cagr_real * 100:+.2f}%"),
            ("Investment Time-Weighted Return (TWR CAGR)", f"{self.twr_cagr_nominal * 100:+.2f}%"),
            ("Money-Weighted Return (MWR / XIRR)", mwr_str),
            ("Gross Wealth Ratio CAGR (Includes Deposits)", f"{self.cagr_nominal * 100:+.2f}%"),
            ("Max Portfolio Drawdown", f"{self.max_drawdown * 100:.2f}%"),
            ("Total Transaction Costs Incurred", f"{sym}{self.total_transaction_costs:,.2f}"),
            ("Rebalancing Events (Sales Executed)", f"{self.rebalance_events_count} times"),
            ("Tax-Efficient Inflow Steering", rebal_mode_str),
            ("Backtest Duration", f"{self.duration_years:.1f} years ({len(self.history_df)} days)")
        ]

        if self.inflow_rebalancing_used and self.estimated_tax_savings > 0:
            summary_rows.append(("Estimated Tax & Friction Savings", f"{sym}{self.estimated_tax_savings:,.2f}"))

        metrics, values = zip(*summary_rows)
        return pd.DataFrame({"Metric": metrics, "Value": values}).set_index("Metric")


def _calculate_xirr(cash_flows: List[Tuple[float, float]]) -> Optional[float]:
    """
    Calculate annualized Money-Weighted Return (XIRR).
    cash_flows is a list of tuples: (time_in_years, amount)
    where cash outflows (investments) are negative and inflows (ending wealth) are positive.
    """
    if len(cash_flows) < 2:
        return None

    def npv(rate: float) -> float:
        if rate <= -0.9999:
            return float("inf")
        val = 0.0
        for t_yr, cf in cash_flows:
            val += cf / ((1.0 + rate) ** t_yr)
        return val

    try:
        cfs = [cf for _, cf in cash_flows]
        has_pos = any(c > 0 for c in cfs)
        has_neg = any(c < 0 for c in cfs)
        if not (has_pos and has_neg):
            return None

        f_low = npv(-0.90)
        f_high = npv(5.0)
        if f_low * f_high < 0:
            sol = optimize.brentq(npv, -0.90, 5.0, maxiter=200)
            return float(sol)
        elif f_low * npv(20.0) < 0:
            sol = optimize.brentq(npv, -0.90, 20.0, maxiter=200)
            return float(sol)
        else:
            sol = optimize.root_scalar(npv, x0=0.08, method="newton")
            if sol.converged and sol.root > -0.99:
                return float(sol.root)
            return None
    except Exception:
        return None


class Backtester:
    """
    Sequential Event-Driven Backtesting Simulator.
    Simulates portfolio cashflows, asset appreciation, rebalancing, tax drag, and fee drag
    day-by-day with zero look-ahead bias and institutional performance attribution.
    """

    def run(self, portfolio: Portfolio, config: Optional[BacktestConfig] = None) -> BacktestResult:
        if portfolio.asset_returns is None or portfolio.asset_returns.empty:
            raise RuntimeError("Portfolio has no return data. Call portfolio.load_data() before backtesting.")

        cfg = config or BacktestConfig()
        asset_rets = portfolio.asset_returns.copy()
        target_weights = portfolio.config.get_top_level_allocations()
        assets = list(target_weights.keys())

        total_w = sum(target_weights.values())
        norm_weights = {k: v / total_w for k, v in target_weights.items()}

        dates = asset_rets.index
        n_days = len(dates)
        if n_days < 2:
            raise ValueError("Insufficient date history to run backtest.")

        start_date = dates[0]
        end_date = dates[-1]
        duration_years = (end_date - start_date).days / 365.25

        init_fee = cfg.initial_capital * cfg.transaction_cost_pct
        current_asset_values = {
            a: norm_weights[a] * (cfg.initial_capital - init_fee) for a in assets
        }
        cumulative_fees = init_fee
        cumulative_contributions = 0.0
        total_invested = cfg.initial_capital
        rebalance_count = 0
        avoided_turnover = 0.0

        use_inflow_steering = (
            cfg.inflow_rebalancing or
            cfg.rebalance_frequency.lower() == RebalanceFrequency.INFLOW_ONLY.value
        )

        # Cash flow schedule for XIRR: list of (time_in_years, cash_flow)
        xirr_cash_flows: List[Tuple[float, float]] = [(0.0, -cfg.initial_capital)]

        # Time-weighted return tracker
        daily_investment_returns: List[float] = []

        history_records = []

        total_nom_0 = sum(current_asset_values.values())
        rec_0 = {
            "Date": start_date,
            "Nominal_Wealth": total_nom_0,
            "Real_Wealth": total_nom_0,
            "Total_Invested_Capital": total_invested,
            "Cumulative_Contributions": cumulative_contributions,
            "Cumulative_Gains": total_nom_0 - total_invested,
            "Cumulative_Fees": cumulative_fees,
            "Cash_Inflow": 0.0,
            "Investment_Return": 0.0,
            "Rebalanced": False,
        }
        for a in assets:
            rec_0[f"Value_{a}"] = current_asset_values[a]
            rec_0[f"Weight_{a}"] = current_asset_values[a] / total_nom_0 if total_nom_0 > 0 else 0.0
        history_records.append(rec_0)

        prev_day_wealth = total_nom_0
        freq_str = cfg.rebalance_frequency.lower()

        for t in range(1, n_days):
            curr_date = dates[t]
            prev_date = dates[t - 1]

            # 1. Asset Price Appreciation (Market movement)
            for a in assets:
                day_ret = asset_rets.loc[curr_date, a]
                current_asset_values[a] *= (1.0 + day_ret)

            # 2. Check Rebalancing Trigger (Before intermediate cash flows)
            total_val_before_rebal = sum(current_asset_values.values())
            should_rebalance = False
            is_new_month = (curr_date.month != prev_date.month)

            if freq_str == RebalanceFrequency.INFLOW_ONLY.value:
                # Inflow-Only mode: Zero physical sales are ever executed
                should_rebalance = False
                if is_new_month and (curr_date.month in (1, 4, 7, 10)) and total_val_before_rebal > 0:
                    avoided_turnover += sum(
                        abs(current_asset_values[a] - norm_weights[a] * total_val_before_rebal)
                        for a in assets
                    ) / 2.0
            elif freq_str == RebalanceFrequency.MONTHLY.value:
                should_rebalance = is_new_month
            elif freq_str == RebalanceFrequency.QUARTERLY.value:
                should_rebalance = is_new_month and (curr_date.month in (1, 4, 7, 10))
            elif freq_str == RebalanceFrequency.SEMI_ANNUAL.value:
                should_rebalance = is_new_month and (curr_date.month in (1, 7))
            elif freq_str == RebalanceFrequency.ANNUAL.value:
                should_rebalance = is_new_month and (curr_date.month == 1)
            elif freq_str == RebalanceFrequency.THRESHOLD.value:
                thresh = cfg.tolerance_band if cfg.tolerance_band is not None else cfg.rebalance_threshold
                for a in assets:
                    current_w = current_asset_values[a] / total_val_before_rebal if total_val_before_rebal > 0 else 0.0
                    if abs(current_w - norm_weights[a]) >= thresh:
                        should_rebalance = True
                        break

            # If tolerance_band is set on calendar rebalancing, check if any asset is outside the band
            if should_rebalance and cfg.tolerance_band is not None and total_val_before_rebal > 0:
                max_dev = max(
                    abs(current_asset_values[a] / total_val_before_rebal - norm_weights[a])
                    for a in assets
                )
                if max_dev < cfg.tolerance_band:
                    # Spares selling because assets are still within momentum corridor
                    should_rebalance = False
                    actual_avoided_turnover = sum(
                        abs(current_asset_values[a] - norm_weights[a] * total_val_before_rebal)
                        for a in assets
                    ) / 2.0
                    avoided_turnover += actual_avoided_turnover

            # Execute Physical Rebalancing (Selling/Buying) if triggered
            if should_rebalance and total_val_before_rebal > 0:
                rebalance_count += 1
                turnover = 0.0
                for a in assets:
                    target_val = norm_weights[a] * total_val_before_rebal
                    turnover += abs(current_asset_values[a] - target_val)
                turnover /= 2.0

                rebal_fee = turnover * cfg.transaction_cost_pct
                # Model capital gains tax on sold assets if configured
                rebal_tax = (turnover * cfg.capital_gains_tax_rate) if cfg.capital_gains_tax_rate > 0 else 0.0
                total_drag = rebal_fee + rebal_tax
                cumulative_fees += total_drag
                total_val_post_rebal = total_val_before_rebal - total_drag

                for a in assets:
                    current_asset_values[a] = norm_weights[a] * total_val_post_rebal

            # 3. Calculate True Daily Investment Return (Pre-contribution)
            pre_contribution_wealth = sum(current_asset_values.values())
            daily_inv_ret = (
                (pre_contribution_wealth - prev_day_wealth) / prev_day_wealth
                if prev_day_wealth > 0 else 0.0
            )
            daily_investment_returns.append(daily_inv_ret)

            # 4. Monthly Periodic Contribution (Cash Inflow Timing: End of Day)
            days_elapsed = (curr_date - start_date).days
            years_elapsed = days_elapsed / 365.25
            inflation_factor = (1.0 + cfg.annual_inflation_rate) ** years_elapsed

            cash_inflow_today = 0.0

            if is_new_month and cfg.monthly_contribution > 0:
                full_years = int(years_elapsed)
                current_monthly_contrib = cfg.monthly_contribution * ((1.0 + cfg.contribution_growth_rate) ** full_years)

                contrib_fee = current_monthly_contrib * cfg.transaction_cost_pct
                net_contrib = current_monthly_contrib - contrib_fee
                cumulative_fees += contrib_fee

                if use_inflow_steering:
                    # --- TAX-EFFICIENT INFLOW REBALANCING (Cash Flow Steering) ---
                    # Direct new capital exclusively to underweight assets relative to post-deposit target
                    pre_total = sum(current_asset_values.values())
                    target_total = pre_total + net_contrib

                    deficits = {}
                    for a in assets:
                        target_val_a = norm_weights[a] * target_total
                        deficits[a] = max(0.0, target_val_a - current_asset_values[a])

                    sum_deficits = sum(deficits.values())
                    if sum_deficits > 0:
                        for a in assets:
                            alloc_a = net_contrib * (deficits[a] / sum_deficits)
                            current_asset_values[a] += alloc_a
                    else:
                        for a in assets:
                            current_asset_values[a] += norm_weights[a] * net_contrib
                else:
                    # Standard naive contribution: split by fixed weights
                    for a in assets:
                        current_asset_values[a] += norm_weights[a] * net_contrib

                total_invested += current_monthly_contrib
                cumulative_contributions += current_monthly_contrib
                cash_inflow_today = current_monthly_contrib
                xirr_cash_flows.append((years_elapsed, -current_monthly_contrib))

            # 5. Snapshot Daily Post-Contribution Performance
            nominal_wealth_t = sum(current_asset_values.values())
            real_wealth_t = nominal_wealth_t / inflation_factor if inflation_factor > 0 else nominal_wealth_t
            prev_day_wealth = nominal_wealth_t

            rec_t = {
                "Date": curr_date,
                "Nominal_Wealth": nominal_wealth_t,
                "Real_Wealth": real_wealth_t,
                "Total_Invested_Capital": total_invested,
                "Cumulative_Contributions": cumulative_contributions,
                "Cumulative_Gains": nominal_wealth_t - total_invested,
                "Cumulative_Fees": cumulative_fees,
                "Cash_Inflow": cash_inflow_today,
                "Investment_Return": daily_inv_ret,
                "Rebalanced": should_rebalance,
            }
            for a in assets:
                rec_t[f"Value_{a}"] = current_asset_values[a]
                rec_t[f"Weight_{a}"] = current_asset_values[a] / nominal_wealth_t if nominal_wealth_t > 0 else 0.0

            history_records.append(rec_t)

        history_df = pd.DataFrame(history_records).set_index("Date")

        final_nominal = history_df["Nominal_Wealth"].iloc[-1]
        final_real = history_df["Real_Wealth"].iloc[-1]
        total_gains = final_nominal - total_invested

        # Append terminal wealth to XIRR cash flow schedule
        xirr_cash_flows.append((duration_years, final_nominal))
        mwr_xirr = _calculate_xirr(xirr_cash_flows)

        # True Time-Weighted Return (TWR)
        inv_ret_series = pd.Series(daily_investment_returns)
        twr_nominal = float((1.0 + inv_ret_series).prod() - 1.0)
        twr_cagr_nom = float((1.0 + twr_nominal) ** (1.0 / duration_years) - 1.0) if duration_years > 0 else 0.0

        # Real TWR (Deflated by continuous inflation)
        total_inflation = (1.0 + cfg.annual_inflation_rate) ** duration_years
        twr_real = float((1.0 + twr_nominal) / total_inflation - 1.0) if total_inflation > 0 else twr_nominal
        twr_cagr_real = float((1.0 + twr_real) ** (1.0 / duration_years) - 1.0) if duration_years > 0 else 0.0

        # Legacy Ratio CAGR: (final / initial)**(1/years) - 1
        cagr_nom = (final_nominal / cfg.initial_capital) ** (1.0 / duration_years) - 1.0 if duration_years > 0 else 0.0
        cagr_real = (final_real / cfg.initial_capital) ** (1.0 / duration_years) - 1.0 if duration_years > 0 else 0.0

        mdd = calculate_max_drawdown(history_df["Nominal_Wealth"], is_returns=False)

        tax_rate_est = max(cfg.transaction_cost_pct, cfg.capital_gains_tax_rate)
        est_tax_savings = avoided_turnover * (cfg.transaction_cost_pct + tax_rate_est)

        return BacktestResult(
            history_df=history_df,
            config=cfg,
            target_weights=norm_weights,
            initial_capital=cfg.initial_capital,
            final_nominal_wealth=final_nominal,
            final_real_wealth=final_real,
            total_contributions=cumulative_contributions,
            total_capital_invested=total_invested,
            total_investment_gains=total_gains,
            total_transaction_costs=cumulative_fees,
            twr_nominal=twr_nominal,
            twr_cagr_nominal=twr_cagr_nom,
            twr_real=twr_real,
            twr_cagr_real=twr_cagr_real,
            mwr_xirr=mwr_xirr,
            cagr_nominal=cagr_nom,
            cagr_real=cagr_real,
            max_drawdown=mdd,
            rebalance_events_count=rebalance_count,
            duration_years=duration_years,
            inflow_rebalancing_used=use_inflow_steering,
            estimated_turnover_avoided=avoided_turnover,
            estimated_tax_savings=est_tax_savings
        )

    def compare_rebalancing_strategies(
        self,
        portfolio: Portfolio,
        base_config: Optional[BacktestConfig] = None
    ) -> pd.DataFrame:
        """
        Compare standard calendar/threshold rebalancing against Tax-Efficient Inflow Rebalancing strategies.
        Quantifies performance, drawdown, rebalancing frequency, and turnover/friction savings.
        """
        cfg = base_config or BacktestConfig()
        sym = cfg.currency_symbol

        strategies = {
            "Quarterly Calendar (Standard)": BacktestConfig(
                initial_capital=cfg.initial_capital,
                monthly_contribution=cfg.monthly_contribution,
                contribution_growth_rate=cfg.contribution_growth_rate,
                rebalance_frequency="quarterly",
                transaction_cost_pct=cfg.transaction_cost_pct,
                annual_inflation_rate=cfg.annual_inflation_rate,
                currency_symbol=sym,
                inflow_rebalancing=False
            ),
            "Threshold 5% Band (Standard)": BacktestConfig(
                initial_capital=cfg.initial_capital,
                monthly_contribution=cfg.monthly_contribution,
                contribution_growth_rate=cfg.contribution_growth_rate,
                rebalance_frequency="threshold",
                rebalance_threshold=0.05,
                transaction_cost_pct=cfg.transaction_cost_pct,
                annual_inflation_rate=cfg.annual_inflation_rate,
                currency_symbol=sym,
                inflow_rebalancing=False
            ),
            "Tax-Efficient Inflow (Quarterly Steering)": BacktestConfig(
                initial_capital=cfg.initial_capital,
                monthly_contribution=cfg.monthly_contribution,
                contribution_growth_rate=cfg.contribution_growth_rate,
                rebalance_frequency="quarterly",
                transaction_cost_pct=cfg.transaction_cost_pct,
                annual_inflation_rate=cfg.annual_inflation_rate,
                currency_symbol=sym,
                inflow_rebalancing=True
            ),
            "Tax-Efficient Inflow (10% Tolerance Corridor)": BacktestConfig(
                initial_capital=cfg.initial_capital,
                monthly_contribution=cfg.monthly_contribution,
                contribution_growth_rate=cfg.contribution_growth_rate,
                rebalance_frequency="threshold",
                rebalance_threshold=0.10,
                tolerance_band=0.10,
                transaction_cost_pct=cfg.transaction_cost_pct,
                annual_inflation_rate=cfg.annual_inflation_rate,
                currency_symbol=sym,
                inflow_rebalancing=True
            ),
            "Inflow-Only (Zero Sales / Pure DCA Steering)": BacktestConfig(
                initial_capital=cfg.initial_capital,
                monthly_contribution=cfg.monthly_contribution,
                contribution_growth_rate=cfg.contribution_growth_rate,
                rebalance_frequency="inflow_only",
                transaction_cost_pct=cfg.transaction_cost_pct,
                annual_inflation_rate=cfg.annual_inflation_rate,
                currency_symbol=sym,
                inflow_rebalancing=True
            )
        }

        rows = []
        for name, strat_cfg in strategies.items():
            res = self.run(portfolio, strat_cfg)
            mwr_str = f"{res.mwr_xirr * 100:+.2f}%" if res.mwr_xirr is not None else "N/A"
            rows.append({
                "Strategy": name,
                "Ending Wealth": f"{sym}{res.final_nominal_wealth:,.0f}",
                "Real Wealth": f"{sym}{res.final_real_wealth:,.0f}",
                "TWR CAGR": f"{res.twr_cagr_nominal * 100:+.2f}%",
                "MWR (XIRR)": mwr_str,
                "Max Drawdown": f"{res.max_drawdown * 100:.2f}%",
                "Sales Executed": f"{res.rebalance_events_count} times",
                "Total Fees Paid": f"{sym}{res.total_transaction_costs:,.2f}",
                "Tax-Efficiency": "HIGH (0 sales)" if res.rebalance_events_count == 0 else ("OPTIMIZED" if "Inflow" in name else "STANDARD")
            })

        return pd.DataFrame(rows).set_index("Strategy")
