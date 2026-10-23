"""
Risk Analytics Module.
Calculates downside volatility, maximum drawdown, peak-to-trough duration,
Sharpe, Sortino, Calmar ratios, higher-order distribution moments (skewness, kurtosis),
parametric, historical, and modified Cornish-Fisher Value at Risk (VaR),
Conditional Value at Risk / Expected Shortfall (CVaR), and marginal / percentage risk contributions.
"""

from typing import Union, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from .returns import (
    calculate_simple_returns,
    calculate_cagr,
    calculate_annualized_return,
    calculate_annualized_volatility,
)


def calculate_downside_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
    target_return: Optional[float] = None,
    ddof: int = 0
) -> Union[float, pd.Series]:
    """
    Calculate annualized downside deviation (semi-deviation below risk-free/target rate).
    sigma_down = sqrt(sum(min(R_t - target_t, 0)^2) / (N - ddof)) * sqrt(periods_per_year).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: calculate_downside_volatility(col.dropna(), risk_free_rate, periods_per_year, target_return, ddof)
        )

    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    target = target_return if target_return is not None else risk_free_rate
    target_per_period = target / periods_per_year
    excess_downside = np.minimum(clean_r - target_per_period, 0.0)

    denominator = max(1, len(clean_r) - ddof)
    downside_var = np.sum(excess_downside ** 2) / denominator
    return float(np.sqrt(downside_var) * np.sqrt(periods_per_year))


def calculate_drawdown_series(
    prices_or_returns: Union[pd.Series, pd.DataFrame],
    is_returns: bool = True
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate time series of percentage drawdown from historical running peak.
    DD_t = (W_t - Peak_t) / Peak_t.
    """
    if is_returns:
        wealth_index = (1.0 + prices_or_returns).cumprod()
    else:
        wealth_index = prices_or_returns

    if isinstance(wealth_index, pd.DataFrame):
        peaks = wealth_index.cummax()
        return (wealth_index - peaks) / peaks
    else:
        peak = wealth_index.cummax()
        return (wealth_index - peak) / peak


def calculate_max_drawdown(
    prices_or_returns: Union[pd.Series, pd.DataFrame],
    is_returns: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Maximum Drawdown (MDD): min(DD_t). Returns a negative float or Series.
    """
    dd = calculate_drawdown_series(prices_or_returns, is_returns=is_returns)
    if isinstance(dd, pd.DataFrame):
        return dd.min()
    clean_dd = dd.dropna()
    return float(clean_dd.min()) if not clean_dd.empty else 0.0


def calculate_drawdown_details(
    prices_or_returns: pd.Series,
    is_returns: bool = True
) -> Dict[str, Any]:
    """
    Calculate peak-to-trough and peak-to-recovery drawdown metrics.
    Returns:
        Dict with max_drawdown, peak_date, trough_date, recovery_date,
        drawdown_duration_days, recovery_duration_days.
    """
    if is_returns:
        wealth = (1.0 + prices_or_returns).cumprod()
    else:
        wealth = prices_or_returns.copy()

    clean_w = wealth.dropna()
    if len(clean_w) < 2:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "drawdown_days": 0,
            "recovery_days": 0
        }

    peaks = clean_w.cummax()
    dd = (clean_w - peaks) / peaks
    mdd = float(dd.min())

    trough_idx = dd.idxmin()
    # Find the peak prior to trough
    peak_series = clean_w.loc[:trough_idx]
    peak_idx = peak_series.idxmax()
    peak_val = clean_w.loc[peak_idx]

    # Find recovery date after trough (when wealth >= peak_val)
    recovery_series = clean_w.loc[trough_idx:]
    recovered_mask = recovery_series >= peak_val
    recovery_idx = recovery_series[recovered_mask].index[0] if recovered_mask.any() else None

    dd_days = (trough_idx - peak_idx).days if hasattr(trough_idx, "days") or isinstance(trough_idx, pd.Timestamp) else 0
    rec_days = (recovery_idx - trough_idx).days if recovery_idx is not None and isinstance(recovery_idx, pd.Timestamp) else None

    return {
        "max_drawdown": mdd,
        "peak_date": peak_idx.strftime("%Y-%m-%d") if isinstance(peak_idx, pd.Timestamp) else str(peak_idx),
        "trough_date": trough_idx.strftime("%Y-%m-%d") if isinstance(trough_idx, pd.Timestamp) else str(trough_idx),
        "recovery_date": recovery_idx.strftime("%Y-%m-%d") if isinstance(recovery_idx, pd.Timestamp) else None,
        "drawdown_days": dd_days,
        "recovery_days": rec_days
    }


def calculate_sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate annualized Sharpe Ratio.
    If geometric=True, uses compound geometric annualized return: (Ann_Geom_Ret - Rf) / Ann_Vol.
    If geometric=False (standard institutional), uses arithmetic mean return: (Ann_Arith_Ret - Rf) / Ann_Vol.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: calculate_sharpe_ratio(col.dropna(), risk_free_rate, periods_per_year, geometric)
        )

    clean_r = returns.dropna()
    ann_ret = calculate_annualized_return(clean_r, periods_per_year=periods_per_year, geometric=geometric)
    ann_vol = calculate_annualized_volatility(clean_r, periods_per_year=periods_per_year)

    if ann_vol <= 1e-8:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def calculate_sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
    target_return: Optional[float] = None,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate annualized Sortino Ratio: (Annualized Return - Target Return) / Annualized Downside Volatility.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: calculate_sortino_ratio(col.dropna(), risk_free_rate, periods_per_year, target_return, geometric)
        )

    clean_r = returns.dropna()
    target = target_return if target_return is not None else risk_free_rate
    ann_ret = calculate_annualized_return(clean_r, periods_per_year=periods_per_year, geometric=geometric)
    downside_vol = calculate_downside_volatility(
        clean_r,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        target_return=target
    )

    if downside_vol <= 1e-8:
        return 0.0
    return float((ann_ret - target) / downside_vol)


def calculate_calmar_ratio(
    prices_or_returns: Union[pd.Series, pd.DataFrame],
    is_returns: bool = True,
    periods_per_year: float = 252.0
) -> Union[float, pd.Series]:
    """
    Calculate Calmar Ratio: CAGR / |Max Drawdown|.
    """
    if isinstance(prices_or_returns, pd.DataFrame):
        return prices_or_returns.apply(lambda col: calculate_calmar_ratio(col.dropna(), is_returns, periods_per_year))

    if is_returns:
        clean_r = prices_or_returns.dropna()
        wealth = (1.0 + clean_r).cumprod()
        cagr = calculate_cagr(wealth, periods_per_year=periods_per_year)
        mdd = abs(calculate_max_drawdown(clean_r, is_returns=True))
    else:
        clean_p = prices_or_returns.dropna()
        cagr = calculate_cagr(clean_p, periods_per_year=periods_per_year)
        mdd = abs(calculate_max_drawdown(clean_p, is_returns=False))

    if mdd <= 1e-8:
        return 0.0
    return float(cagr / mdd)


def calculate_skewness(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate Fisher-Pearson standardized moment of skewness.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: calculate_skewness(col.dropna()))
    clean_r = returns.dropna()
    if len(clean_r) < 3:
        return 0.0
    return float(stats.skew(clean_r, bias=False))


def calculate_kurtosis(returns: Union[pd.Series, pd.DataFrame], excess: bool = True) -> Union[float, pd.Series]:
    """
    Calculate Fisher excess kurtosis (normal distribution = 0).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: calculate_kurtosis(col.dropna(), excess=excess))
    clean_r = returns.dropna()
    if len(clean_r) < 4:
        return 0.0
    kurt = stats.kurtosis(clean_r, fisher=excess, bias=False)
    return float(kurt)


def calculate_var(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_level: float = 0.95,
    method: str = "historical",
    return_loss_as_positive: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Value at Risk (VaR).
    By default (return_loss_as_positive=True), returns a positive number representing loss:
    e.g. +0.02 means 95% of returns exceed -2.0%.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: calculate_var(col.dropna(), confidence_level, method, return_loss_as_positive)
        )

    clean_r = returns.dropna()
    if len(clean_r) < 5:
        return 0.0

    alpha = 1.0 - confidence_level

    if method == "historical":
        raw_cutoff = float(np.percentile(clean_r, alpha * 100.0))
        return -raw_cutoff if return_loss_as_positive else raw_cutoff
    elif method == "parametric":
        mu = clean_r.mean()
        sigma = clean_r.std(ddof=1)
        z = stats.norm.ppf(alpha)
        raw_cutoff = float(mu + z * sigma)
        return -raw_cutoff if return_loss_as_positive else raw_cutoff
    elif method == "modified":
        mu = clean_r.mean()
        sigma = clean_r.std(ddof=1)
        s = calculate_skewness(clean_r)
        k = calculate_kurtosis(clean_r, excess=True)
        z = stats.norm.ppf(alpha)
        z_cf = (
            z +
            (z**2 - 1) * s / 6.0 +
            (z**3 - 3*z) * k / 24.0 -
            (2*z**3 - 5*z) * (s**2) / 36.0
        )
        raw_cutoff = float(mu + z_cf * sigma)
        return -raw_cutoff if return_loss_as_positive else raw_cutoff
    else:
        raise ValueError(f"Unknown VaR method: '{method}'. Choose 'historical', 'parametric', or 'modified'.")


def calculate_cvar(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_level: float = 0.95,
    method: str = "historical",
    return_loss_as_positive: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda col: calculate_cvar(col.dropna(), confidence_level, method, return_loss_as_positive)
        )

    clean_r = returns.dropna()
    if len(clean_r) < 5:
        return 0.0

    var_loss = calculate_var(clean_r, confidence_level=confidence_level, method=method, return_loss_as_positive=True)
    cutoff = -var_loss
    tail_losses = clean_r[clean_r <= cutoff]

    if len(tail_losses) == 0:
        return var_loss if return_loss_as_positive else cutoff

    avg_tail = float(tail_losses.mean())
    return -avg_tail if return_loss_as_positive else avg_tail


def calculate_risk_contributions(
    weights: Union[Dict[str, float], pd.Series, np.ndarray],
    cov_matrix: Union[pd.DataFrame, np.ndarray]
) -> pd.DataFrame:
    """
    Calculate Marginal Contribution to Risk (MCR), Total Risk Contribution (TRC),
    and Percentage Risk Contribution (%RC) for each asset.
    Sigma_p = sqrt(w^T * Cov * w)
    MCR_i = (Cov * w)_i / Sigma_p
    TRC_i = w_i * MCR_i
    %RC_i = TRC_i / Sigma_p
    """
    if isinstance(weights, dict):
        asset_names = list(weights.keys())
        w = np.array([weights[a] for a in asset_names], dtype=float)
    elif isinstance(weights, pd.Series):
        asset_names = list(weights.index)
        w = weights.values.astype(float)
    else:
        w = np.array(weights, dtype=float)
        asset_names = [f"Asset_{i}" for i in range(len(w))]

    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.loc[asset_names, asset_names].values
    else:
        cov = np.array(cov_matrix, dtype=float)

    total_w = np.sum(w)
    if total_w > 0:
        w = w / total_w

    port_var = float(w.T @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-8

    # Marginal Risk Contribution
    mcr = (cov @ w) / port_vol
    # Absolute Total Risk Contribution
    trc = w * mcr
    # Percentage Risk Contribution
    prc = trc / port_vol

    df = pd.DataFrame({
        "Weight (%)": w * 100.0,
        "Marginal Vol Contribution": mcr,
        "Absolute Risk Contribution": trc,
        "Percentage Risk Contribution (%)": prc * 100.0
    }, index=asset_names)
    return df


def calculate_annual_calendar_returns(
    returns: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute total calendar year returns for each year in the data.
    """
    if isinstance(returns, pd.Series):
        df_rets = returns.to_frame()
    else:
        df_rets = returns.copy()

    if not isinstance(df_rets.index, pd.DatetimeIndex):
        raise TypeError("Returns must have a DatetimeIndex to calculate annual calendar returns.")

    annual_df = df_rets.groupby(df_rets.index.year).apply(lambda yr_df: (1.0 + yr_df).prod() - 1.0)
    annual_df.index.name = "Year"
    return annual_df


def compute_comprehensive_risk_report(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0
) -> pd.DataFrame:
    """
    Compute a complete institutional-grade risk & distribution analytics table.
    """
    summary_rows = []
    annual_rets = calculate_annual_calendar_returns(returns)

    for col in returns.columns:
        r_series = returns[col].dropna()
        ann_ret = calculate_annualized_return(r_series, periods_per_year, geometric=True)
        cagr = calculate_cagr((1.0 + r_series).cumprod(), periods_per_year)
        ann_vol = calculate_annualized_volatility(r_series, periods_per_year)
        down_vol = calculate_downside_volatility(r_series, risk_free_rate, periods_per_year)
        mdd = calculate_max_drawdown(r_series, is_returns=True)
        sharpe = calculate_sharpe_ratio(r_series, risk_free_rate, periods_per_year)
        sortino = calculate_sortino_ratio(r_series, risk_free_rate, periods_per_year)
        calmar = calculate_calmar_ratio(r_series, is_returns=True, periods_per_year=periods_per_year)
        skew = calculate_skewness(r_series)
        kurt = calculate_kurtosis(r_series, excess=True)
        var_95 = calculate_var(r_series, confidence_level=0.95, method="historical")
        cvar_95 = calculate_cvar(r_series, confidence_level=0.95, method="historical")
        var_99 = calculate_var(r_series, confidence_level=0.99, method="historical")
        cvar_99 = calculate_cvar(r_series, confidence_level=0.99, method="historical")

        col_annual = annual_rets[col].dropna()
        best_yr = float(col_annual.max()) if not col_annual.empty else 0.0
        worst_yr = float(col_annual.min()) if not col_annual.empty else 0.0

        summary_rows.append({
            "Asset": col,
            "Ann. Return (%)": ann_ret * 100.0,
            "CAGR (%)": cagr * 100.0,
            "Ann. Volatility (%)": ann_vol * 100.0,
            "Downside Vol (%)": down_vol * 100.0,
            "Max Drawdown (%)": mdd * 100.0,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Skewness": skew,
            "Excess Kurtosis": kurt,
            "Daily VaR 95% (%)": var_95 * 100.0,
            "Daily CVaR 95% (%)": cvar_95 * 100.0,
            "Daily VaR 99% (%)": var_99 * 100.0,
            "Daily CVaR 99% (%)": cvar_99 * 100.0,
            "Best Year (%)": best_yr * 100.0,
            "Worst Year (%)": worst_yr * 100.0
        })

    return pd.DataFrame(summary_rows).set_index("Asset")
