"""
Analytics package: Return calculations, risk metrics, validation, and performance analytics.
"""

from .returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
    calculate_cagr,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_portfolio_returns,
    compute_return_summary,
)
from .risk import (
    calculate_downside_volatility,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_drawdown_details,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_skewness,
    calculate_kurtosis,
    calculate_var,
    calculate_cvar,
    calculate_risk_contributions,
    calculate_annual_calendar_returns,
    compute_comprehensive_risk_report,
)
from .correlation import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_rolling_correlation,
    calculate_rolling_volatility,
    format_correlation_heatmap_text,
)
from .validation import (
    WalkForwardValidator,
    StressTestEngine,
)

__all__ = [
    # Returns
    "calculate_simple_returns",
    "calculate_log_returns",
    "calculate_cumulative_returns",
    "calculate_cagr",
    "calculate_annualized_return",
    "calculate_annualized_volatility",
    "calculate_portfolio_returns",
    "compute_return_summary",
    # Risk
    "calculate_downside_volatility",
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_drawdown_details",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_skewness",
    "calculate_kurtosis",
    "calculate_var",
    "calculate_cvar",
    "calculate_risk_contributions",
    "calculate_annual_calendar_returns",
    "compute_comprehensive_risk_report",
    # Correlation
    "calculate_correlation_matrix",
    "calculate_covariance_matrix",
    "calculate_rolling_correlation",
    "calculate_rolling_volatility",
    "format_correlation_heatmap_text",
    # Validation & Stress Testing
    "WalkForwardValidator",
    "StressTestEngine",
]
