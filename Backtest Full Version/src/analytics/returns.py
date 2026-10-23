"""
Return Analytics Module.
Calculates historical arithmetic, log, cumulative, annualized returns, CAGR,
annualized volatility, and multi-asset portfolio returns (rebalanced and drift).
"""

from typing import Union, Dict, Optional
import numpy as np
import pandas as pd


def calculate_simple_returns(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate discrete arithmetic percentage returns: R_t = (P_t - P_{t-1}) / P_{t-1}.
    """
    return prices.pct_change().dropna(how="all")


def calculate_log_returns(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate continuously compounded log returns: r_t = ln(P_t / P_{t-1}).
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


def calculate_cumulative_returns(returns: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate cumulative compounding returns from a series of period returns:
    R_cum,t = prod(1 + R_i) - 1.
    """
    return (1.0 + returns).cumprod() - 1.0


def calculate_cagr(
    prices: Union[pd.Series, pd.DataFrame],
    periods_per_year: float = 252.0
) -> Union[float, pd.Series]:
    """
    Calculate Compound Annual Growth Rate (CAGR) from price time series.

    If DataFrame/Series has a DatetimeIndex spanning at least 1 day, uses exact year fraction:
    CAGR = (P_end / P_start) ** (365.25 / calendar_days) - 1.
    Otherwise uses period count:
    CAGR = (P_end / P_start) ** (periods_per_year / n_periods) - 1.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(lambda col: calculate_cagr(col.dropna(), periods_per_year))

    clean_s = prices.dropna()
    if len(clean_s) < 2:
        return 0.0

    p_start = clean_s.iloc[0]
    p_end = clean_s.iloc[-1]

    if p_start <= 0 or p_end <= 0:
        return 0.0

    if isinstance(clean_s.index, pd.DatetimeIndex):
        days = (clean_s.index[-1] - clean_s.index[0]).days
        if days > 0:
            years = days / 365.25
            return float((p_end / p_start) ** (1.0 / years) - 1.0)

    n_periods = len(clean_s) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((p_end / p_start) ** (1.0 / years) - 1.0)


def calculate_annualized_return(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: float = 252.0,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate annualized return from periodic return series.

    Args:
        returns: Return series or DataFrame.
        periods_per_year: 252 for daily, 52 for weekly, 12 for monthly.
        geometric: If True, calculates compound geometric annualized return;
                   if False, calculates mean arithmetic annualized return.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: calculate_annualized_return(col.dropna(), periods_per_year, geometric))

    clean_r = returns.dropna()
    n = len(clean_r)
    if n == 0:
        return 0.0

    if geometric:
        total_growth = (1.0 + clean_r).prod()
        if total_growth <= 0:
            return -1.0
        return float(total_growth ** (periods_per_year / n) - 1.0)
    else:
        return float(clean_r.mean() * periods_per_year)


def calculate_annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: float = 252.0
) -> Union[float, pd.Series]:
    """
    Calculate annualized standard deviation of returns: sigma * sqrt(periods_per_year).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: calculate_annualized_volatility(col.dropna(), periods_per_year))

    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    return float(clean_r.std(ddof=1) * np.sqrt(periods_per_year))


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Union[Dict[str, float], pd.Series],
    rebalanced: bool = True
) -> pd.Series:
    """
    Calculate time series of portfolio returns given asset return matrix and asset weights.

    Args:
        asset_returns: DataFrame where columns are asset/ticker names and rows are period returns.
        weights: Dictionary or Series mapping asset name to target allocation.
        rebalanced: If True, assumes fixed target weights rebalanced at every period.
                    If False (buy & hold), tracks asset value drift over time.

    Returns:
        pd.Series: Portfolio periodic return time series.
    """
    if asset_returns.empty:
        return pd.Series(dtype=float)

    available_assets = [col for col in asset_returns.columns if col in weights]
    if not available_assets:
        raise ValueError("None of the specified assets in weights are present in the returns DataFrame.")

    raw_w = np.array([weights[col] for col in available_assets], dtype=float)
    total_w = raw_w.sum()
    if total_w <= 0:
        raise ValueError("Sum of asset weights must be strictly positive.")
    w = raw_w / total_w

    subset_rets = asset_returns[available_assets].copy()

    if rebalanced:
        port_rets = subset_rets.dot(w)
        port_rets.name = "Portfolio"
        return port_rets
    else:
        cum_asset_growth = (1.0 + subset_rets).cumprod()
        initial_weighted_values = w
        portfolio_values = (cum_asset_growth * initial_weighted_values).sum(axis=1)

        shifted_values = portfolio_values.shift(1)
        shifted_values.iloc[0] = 1.0
        port_rets = (portfolio_values - shifted_values) / shifted_values
        port_rets.name = "Portfolio"
        return port_rets


def compute_return_summary(
    prices_or_returns: Union[pd.DataFrame, pd.Series],
    is_returns: bool = False,
    periods_per_year: float = 252.0
) -> pd.DataFrame:
    """
    Produce a clean summary table of key return metrics across all provided assets.
    """
    if is_returns:
        returns_df = prices_or_returns.to_frame() if isinstance(prices_or_returns, pd.Series) else prices_or_returns
        prices_df = (1.0 + returns_df).cumprod()
    else:
        prices_df = prices_or_returns.to_frame() if isinstance(prices_or_returns, pd.Series) else prices_or_returns
        returns_df = calculate_simple_returns(prices_df)

    summary_rows = []
    for col in prices_df.columns:
        p_series = prices_df[col].dropna()
        r_series = returns_df[col].dropna()

        tot_return = float((p_series.iloc[-1] / p_series.iloc[0]) - 1.0) if len(p_series) >= 2 else 0.0
        cagr = calculate_cagr(p_series, periods_per_year)
        ann_return = calculate_annualized_return(r_series, periods_per_year, geometric=True)
        ann_vol = calculate_annualized_volatility(r_series, periods_per_year)

        summary_rows.append({
            "Asset": col,
            "Total Return (%)": tot_return * 100.0,
            "CAGR (%)": cagr * 100.0,
            "Ann. Geometric Return (%)": ann_return * 100.0,
            "Ann. Volatility (%)": ann_vol * 100.0,
            "Observations": len(r_series)
        })

    return pd.DataFrame(summary_rows).set_index("Asset")