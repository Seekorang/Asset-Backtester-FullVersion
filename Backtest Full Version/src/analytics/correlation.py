"""
Correlation & Covariance Analytics Module.
Computes Pearson correlation matrix, annualized covariance matrix, rolling pairwise
correlations, rolling annualized volatilities, terminal text heatmaps,
and Dynamic Regime-Switching Correlation (Normal vs Crisis market states).
"""

from dataclasses import dataclass, field
from typing import Union, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class RegimeCorrelationResult:
    """
    Structured results from Regime-Switching Correlation analysis.
    Captures asymmetric correlation breakdown and shifts between Normal and Crisis regimes.
    """
    normal_corr: pd.DataFrame
    crisis_corr: pd.DataFrame
    full_sample_corr: pd.DataFrame
    correlation_shift: pd.DataFrame  # crisis_corr - normal_corr (measures contagion/breakdown)
    normal_cov: pd.DataFrame
    crisis_cov: pd.DataFrame
    regime_labels: pd.Series         # Series mapping dates to 'Normal' or 'Crisis'
    regime_counts: Dict[str, int]
    method: str
    threshold_value: float
    description: str

    @property
    def average_normal_correlation(self) -> float:
        """Mean off-diagonal pairwise correlation in normal regime."""
        vals = self.normal_corr.values
        n = vals.shape[0]
        if n <= 1:
            return 1.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(vals[mask]))

    @property
    def average_crisis_correlation(self) -> float:
        """Mean off-diagonal pairwise correlation in crisis regime."""
        vals = self.crisis_corr.values
        n = vals.shape[0]
        if n <= 1:
            return 1.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(vals[mask]))

    @property
    def average_correlation_shift(self) -> float:
        """Average increase in pairwise correlation during stress (correlation breakdown)."""
        return self.average_crisis_correlation - self.average_normal_correlation

    def summary_text(self) -> str:
        """Format an institutional comparative report."""
        assets = list(self.full_sample_corr.columns)
        n_assets = len(assets)
        lines = [
            "=" * 78,
            "  REGIME-SWITCHING CORRELATION & DIVERSIFICATION BREAKDOWN AUDIT",
            "=" * 78,
            f"Segmentation Method : {self.method} (Threshold: {self.threshold_value:.4f})",
            f"Description          : {self.description}",
            f"Sample Observations  : Normal: {self.regime_counts.get('Normal', 0):,} days | Crisis: {self.regime_counts.get('Crisis', 0):,} days",
            "-" * 78,
            "1. NORMAL / CALM REGIME CORRELATION MATRIX (Diversified State):",
            f"{'Asset':<10}" + "".join(f"{a:>10}" for a in assets),
            "-" * (10 + 10 * n_assets)
        ]
        for row in assets:
            row_str = f"{row:<10}" + "".join(f"{self.normal_corr.loc[row, col]:>10.2f}" for col in assets)
            lines.append(row_str)

        lines.extend([
            "-" * 78,
            "2. CRISIS / STRESS REGIME CORRELATION MATRIX (Liquidity Shock / Crash State):",
            f"{'Asset':<10}" + "".join(f"{a:>10}" for a in assets),
            "-" * (10 + 10 * n_assets)
        ])
        for row in assets:
            row_str = f"{row:<10}" + "".join(f"{self.crisis_corr.loc[row, col]:>10.2f}" for col in assets)
            lines.append(row_str)

        lines.extend([
            "-" * 78,
            "3. CORRELATION SHIFT MATRIX (Delta = Crisis - Normal | Positive = Contagion):",
            f"{'Asset':<10}" + "".join(f"{a:>10}" for a in assets),
            "-" * (10 + 10 * n_assets)
        ])
        for row in assets:
            row_str = f"{row:<10}" + "".join(f"{self.correlation_shift.loc[row, col]:>+10.2f}" for col in assets)
            lines.append(row_str)

        lines.extend([
            "-" * 78,
            "4. QUANTITATIVE DIVERSIFICATION & RISK CONTAMINATION TAKEAWAYS:",
            f"  * Average Pairwise Correlation in Normal State : {self.average_normal_correlation:+.3f}",
            f"  * Average Pairwise Correlation in Crisis State : {self.average_crisis_correlation:+.3f}",
            f"  * Overall Diversification Breakdown (Delta)   : {self.average_correlation_shift:+.3f} points",
        ])

        if self.average_correlation_shift > 0.10:
            lines.append("  [!] WARNING: Severe correlation contagion detected during crisis regimes.")
            lines.append("      Cross-asset diversification degrades significantly under market stress.")
        elif self.average_correlation_shift < 0.0:
            lines.append("  [*] BENEFIT: Portfolio displays flight-to-safety characteristics (decorrelation under stress).")
        else:
            lines.append("  [*] Stable cross-asset relationships across calm and stressed market regimes.")

        lines.append("=" * 78)
        return "\n".join(lines)


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate empirical Pearson correlation matrix across asset return series.
    """
    clean_rets = returns.dropna(how="all")
    return clean_rets.corr(method="pearson")


def calculate_covariance_matrix(returns: pd.DataFrame, periods_per_year: float = 252.0) -> pd.DataFrame:
    """
    Calculate annualized covariance matrix: Cov(R_i, R_j) * periods_per_year.
    """
    clean_rets = returns.dropna(how="all")
    return clean_rets.cov() * periods_per_year


def calculate_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Calculate rolling Pearson correlation between two asset return series over a rolling window.
    """
    df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    rolling_corr = df["a"].rolling(window=window).corr(df["b"]).dropna()

    name_a = str(series_a.name) if series_a.name is not None else "AssetA"
    name_b = str(series_b.name) if series_b.name is not None else "AssetB"
    rolling_corr.name = f"Rolling_Corr_{name_a}_{name_b}"
    return rolling_corr


def calculate_rolling_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 60,
    periods_per_year: float = 252.0
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate rolling annualized standard deviation of returns: rolling_std * sqrt(periods_per_year).
    """
    clean_rets = returns.dropna(how="all")
    return clean_rets.rolling(window=window).std() * np.sqrt(periods_per_year)


def calculate_regime_switching_correlation(
    returns: pd.DataFrame,
    method: str = "volatility_threshold",
    threshold: float = 75.0,
    benchmark_col: Optional[str] = None,
    rolling_window: int = 21,
    periods_per_year: float = 252.0
) -> RegimeCorrelationResult:
    """
    Model asymmetric Regime-Switching Correlation across Normal and Crisis market states.

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset return time series.
    method : str
        Regime classification method:
        - 'volatility_threshold': Days where rolling annualized vol exceeds threshold percentile (e.g. 75th).
        - 'market_drawdown': Days where benchmark drawdown is deeper than threshold (e.g. -0.10 or 10%).
        - 'tail_loss': Days where benchmark return is in the bottom tail (e.g. 15th percentile).
    threshold : float
        Threshold cutoff for regime classification.
    benchmark_col : Optional[str]
        Asset column to evaluate market regime. If None, uses equal-weighted average return.
    rolling_window : int
        Window for rolling volatility or drawdown smoothing.
    periods_per_year : float
        Annualization factor (252 for daily).

    Returns:
    --------
    RegimeCorrelationResult
    """
    clean_rets = returns.dropna(how="all").copy()
    if clean_rets.empty or len(clean_rets) < 10:
        raise ValueError("Insufficient return observations for regime-switching correlation analysis.")

    # 1. Identify Benchmark or Composite Market Series
    if benchmark_col and benchmark_col in clean_rets.columns:
        market_series = clean_rets[benchmark_col]
        bench_desc = benchmark_col
    else:
        market_series = clean_rets.mean(axis=1)
        bench_desc = "Equal-Weighted Basket"

    method_lower = method.lower()
    regime_labels = pd.Series(index=clean_rets.index, dtype=str)

    if method_lower == "volatility_threshold":
        rolling_vol = market_series.rolling(window=rolling_window).std() * np.sqrt(periods_per_year)
        valid_vol = rolling_vol.dropna()
        if valid_vol.empty:
            vol_cutoff = 0.20
        else:
            vol_cutoff = float(np.percentile(valid_vol, threshold))

        # Flag crisis when rolling volatility exceeds the percentile cutoff
        is_crisis = (rolling_vol > vol_cutoff)
        regime_labels[is_crisis] = "Crisis"
        regime_labels[~is_crisis] = "Normal"
        threshold_val = vol_cutoff
        desc = f"Rolling {rolling_window}-day vol of '{bench_desc}' > {threshold}th percentile ({vol_cutoff*100:.1f}% ann.)"

    elif method_lower == "market_drawdown":
        wealth_idx = (1.0 + market_series).cumprod()
        peak = wealth_idx.cummax()
        dd = (wealth_idx - peak) / peak
        dd_cutoff = -abs(threshold / 100.0) if threshold > 1.0 else -abs(threshold)

        is_crisis = (dd < dd_cutoff)
        regime_labels[is_crisis] = "Crisis"
        regime_labels[~is_crisis] = "Normal"
        threshold_val = dd_cutoff
        desc = f"Cumulative drawdown of '{bench_desc}' deeper than {dd_cutoff*100:.1f}%"

    elif method_lower == "tail_loss":
        cutoff_ret = float(np.percentile(market_series, threshold))
        is_crisis = (market_series <= cutoff_ret)
        regime_labels[is_crisis] = "Crisis"
        regime_labels[~is_crisis] = "Normal"
        threshold_val = cutoff_ret
        desc = f"Single-day return of '{bench_desc}' in lowest {threshold:.1f}% tail (<= {cutoff_ret*100:.2f}%)"

    else:
        raise ValueError(f"Unknown regime classification method '{method}'. Choose 'volatility_threshold', 'market_drawdown', or 'tail_loss'.")

    # Drop any initial unclassified periods (due to rolling windows)
    regime_labels = regime_labels.dropna()
    aligned_rets = clean_rets.loc[regime_labels.index]

    normal_rets = aligned_rets[regime_labels == "Normal"]
    crisis_rets = aligned_rets[regime_labels == "Crisis"]

    full_corr = clean_rets.corr(method="pearson")

    # Fallback to full correlation if a regime has fewer than 3 samples
    if len(normal_rets) >= 3:
        normal_corr = normal_rets.corr(method="pearson")
        normal_cov = normal_rets.cov() * periods_per_year
    else:
        normal_corr = full_corr.copy()
        normal_cov = clean_rets.cov() * periods_per_year

    if len(crisis_rets) >= 3:
        crisis_corr = crisis_rets.corr(method="pearson")
        crisis_cov = crisis_rets.cov() * periods_per_year
    else:
        crisis_corr = full_corr.copy()
        crisis_cov = clean_rets.cov() * periods_per_year

    # Correlation shift: Crisis minus Normal (contagion)
    corr_shift_df = crisis_corr - normal_corr
    shift_vals = corr_shift_df.to_numpy().copy()
    np.fill_diagonal(shift_vals, 0.0)
    corr_shift = pd.DataFrame(shift_vals, index=full_corr.index, columns=full_corr.columns)

    regime_counts = {
        "Normal": int((regime_labels == "Normal").sum()),
        "Crisis": int((regime_labels == "Crisis").sum())
    }

    return RegimeCorrelationResult(
        normal_corr=normal_corr,
        crisis_corr=crisis_corr,
        full_sample_corr=full_corr,
        correlation_shift=corr_shift,
        normal_cov=normal_cov,
        crisis_cov=crisis_cov,
        regime_labels=regime_labels,
        regime_counts=regime_counts,
        method=method,
        threshold_value=threshold_val,
        description=desc
    )


def format_correlation_heatmap_text(corr_matrix: pd.DataFrame) -> str:
    """
    Generate an ASCII / text heatmap table representing the correlation matrix.
    """
    assets = list(corr_matrix.columns)
    lines = [
        "================ CORRELATION MATRIX ================",
        f"{'Asset':<10}" + "".join(f"{a:>10}" for a in assets),
        "-" * (10 + 10 * len(assets))
    ]
    for row in assets:
        row_str = f"{row:<10}"
        for col in assets:
            val = corr_matrix.loc[row, col]
            row_str += f"{val:>10.2f}"
        lines.append(row_str)

    lines.append("-" * (10 + 10 * len(assets)))
    lines.append("Legend: 1.00 (Perfect), 0.7-0.9 (Strong Pos), 0.3-0.7 (Moderate Pos), -0.3-0.3 (Uncorrelated)")
    lines.append("====================================================")
    return "\n".join(lines)
