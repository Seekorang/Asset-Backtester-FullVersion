"""
Portfolio Engine.
Coordinates data ingestion, multi-currency conversion, basket synthetic indexing,
portfolio level calculations, and unified multi-asset performance, risk, and Monte Carlo tracking.
"""

from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd
import logging

from .allocation import PortfolioConfig, AssetConfig
from ..data.provider import MarketDataProvider
from ..data.validator import DataValidator, DataQualityReport
from ..data.currency import (
    infer_currency_from_ticker,
    convert_asset_prices_to_base,
    convert_asset_returns_to_base,
)
from ..analytics.returns import (
    calculate_simple_returns,
    calculate_portfolio_returns,
    calculate_annualized_return,
    compute_return_summary,
)
from ..analytics.risk import (
    compute_comprehensive_risk_report,
    calculate_drawdown_series,
)
from ..analytics.correlation import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_rolling_correlation,
    calculate_rolling_volatility,
)

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Core Portfolio container and orchestrator.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.raw_prices: Optional[pd.DataFrame] = None
        self.aligned_ticker_prices: Optional[pd.DataFrame] = None
        self.fx_rates: Optional[pd.DataFrame] = None
        self.asset_prices: Optional[pd.DataFrame] = None
        self.asset_returns: Optional[pd.DataFrame] = None
        self.local_asset_returns: Optional[pd.DataFrame] = None
        self.portfolio_returns: Optional[pd.Series] = None
        self.data_quality_report: Optional[DataQualityReport] = None
        self.periods_per_year: float = 252.0

    @property
    def name(self) -> str:
        return self.config.portfolio_name

    @property
    def base_currency(self) -> str:
        return self.config.base_currency

    def get_allocation_breakdown(self) -> pd.DataFrame:
        """Return clear breakdown of top-level asset allocations, currency, and constituent details."""
        rows = []
        for asset_name, asset in self.config.assets.items():
            curr = asset.currency or self.base_currency
            if asset.type == "basket" and asset.constituents:
                constituents_str = ", ".join(f"{t}: {w*100:.1f}%" for t, w in asset.constituents.items())
                rows.append({
                    "Asset": asset_name,
                    "Type": asset.type,
                    "Portfolio Weight (%)": asset.allocation * 100.0,
                    "Quote Currency": curr,
                    "Ticker / Constituents": f"Basket ({constituents_str})",
                    "Description": asset.description or ""
                })
            else:
                rows.append({
                    "Asset": asset_name,
                    "Type": asset.type,
                    "Portfolio Weight (%)": asset.allocation * 100.0,
                    "Quote Currency": curr,
                    "Ticker / Constituents": asset.ticker or "",
                    "Description": asset.description or ""
                })
        return pd.DataFrame(rows)

    def load_data(
        self,
        provider: MarketDataProvider,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        alignment_strategy: Optional[str] = None,
    ) -> DataQualityReport:
        """
        Fetch market data for all required tickers and FX pairs, validate/align,
        and construct multi-currency investor base-currency price and return time series.
        """
        import datetime
        today_date = datetime.date.today()
        today_str = today_date.strftime("%Y-%m-%d")

        s_date = start_date or self.config.market_data_settings.get("start_date", "2018-01-01")
        e_date = end_date or self.config.market_data_settings.get("end_date", today_str)

        if not e_date or str(e_date).strip().lower() in ("today", "now"):
            e_date = today_str
        else:
            try:
                parsed_e = datetime.date.fromisoformat(str(e_date).strip())
                if parsed_e > today_date:
                    logger.info(f"Clamping future end_date '{e_date}' to today: {today_str}")
                    e_date = today_str
            except Exception:
                e_date = today_str
        freq = frequency or self.config.market_data_settings.get("frequency", "daily")
        align = alignment_strategy or self.config.market_data_settings.get("calendar_alignment", "forward_fill")
        outlier_std = float(self.config.market_data_settings.get("outlier_std_threshold", 4.0))

        freq_lower = freq.lower()
        if freq_lower in ["daily", "d"]:
            self.periods_per_year = 252.0
        elif freq_lower in ["weekly", "w"]:
            self.periods_per_year = 52.0
        elif freq_lower in ["monthly", "m"]:
            self.periods_per_year = 12.0

        required_tickers = self.config.get_required_tickers()
        required_fx = self.config.get_required_fx_tickers()
        all_fetch = sorted(list(set(required_tickers + required_fx)))
        logger.info(f"Ingesting market data for assets: {required_tickers} and FX pairs: {required_fx}")

        raw_df = provider.fetch_history(
            tickers=all_fetch,
            start_date=s_date,
            end_date=e_date,
            frequency=freq
        )
        self.raw_prices = raw_df

        # Validate & Align Data
        validator = DataValidator(outlier_std_threshold=outlier_std)
        aligned_df, report = validator.validate_and_align(raw_df, alignment_strategy=align)
        self.aligned_ticker_prices = aligned_df
        self.data_quality_report = report

        # Extract FX rates
        if required_fx:
            available_fx = [fx for fx in required_fx if fx in aligned_df.columns]
            self.fx_rates = aligned_df[available_fx]
        else:
            self.fx_rates = pd.DataFrame(index=aligned_df.index)

        # Get currency mappings for tickers
        ticker_currencies = self.config.get_effective_ticker_currencies()

        # Calculate local raw discrete percentage returns for underlying tickers
        ticker_local_returns = calculate_simple_returns(aligned_df)

        # Convert each ticker's return and price into portfolio base currency
        ticker_base_returns = pd.DataFrame(index=ticker_local_returns.index)
        ticker_base_prices = pd.DataFrame(index=aligned_df.index)

        for t in required_tickers:
            if t in aligned_df.columns:
                t_curr = ticker_currencies.get(t, infer_currency_from_ticker(t))
                ticker_base_returns[t] = convert_asset_returns_to_base(
                    local_returns=ticker_local_returns[t],
                    from_currency=t_curr,
                    base_currency=self.base_currency,
                    fx_rates_df=self.fx_rates
                )
                ticker_base_prices[t] = convert_asset_prices_to_base(
                    prices=aligned_df[t],
                    from_currency=t_curr,
                    base_currency=self.base_currency,
                    fx_rates_df=self.fx_rates
                )
            else:
                raise KeyError(f"Required ticker '{t}' not present in aligned price data.")

        # Construct top-level asset price & return series in Base Currency
        asset_rets_dict: Dict[str, pd.Series] = {}
        asset_local_rets_dict: Dict[str, pd.Series] = {}
        asset_price_dict: Dict[str, pd.Series] = {}

        for asset_name, asset in self.config.assets.items():
            if asset.type == "basket" and asset.constituents:
                basket_base_ret = pd.Series(0.0, index=ticker_base_returns.index)
                basket_local_ret = pd.Series(0.0, index=ticker_local_returns.index)
                for sub_ticker, sub_weight in asset.constituents.items():
                    basket_base_ret += sub_weight * ticker_base_returns[sub_ticker]
                    basket_local_ret += sub_weight * ticker_local_returns[sub_ticker]
                asset_rets_dict[asset_name] = basket_base_ret
                asset_local_rets_dict[asset_name] = basket_local_ret
                asset_price_dict[asset_name] = 100.0 * (1.0 + basket_base_ret).cumprod()
            else:
                ticker = asset.ticker
                if ticker and ticker in ticker_base_returns.columns:
                    asset_rets_dict[asset_name] = ticker_base_returns[ticker]
                    asset_local_rets_dict[asset_name] = ticker_local_returns[ticker]
                    asset_price_dict[asset_name] = ticker_base_prices[ticker]
                else:
                    raise KeyError(f"Ticker '{ticker}' for asset '{asset_name}' missing from price data.")

        self.asset_returns = pd.DataFrame(asset_rets_dict)
        self.local_asset_returns = pd.DataFrame(asset_local_rets_dict)
        self.asset_prices = pd.DataFrame(asset_price_dict)

        # Calculate base rebalanced portfolio return (denominated in Base Currency)
        top_weights = self.config.get_top_level_allocations()
        self.portfolio_returns = calculate_portfolio_returns(
            asset_returns=self.asset_returns,
            weights=top_weights,
            rebalanced=True
        )

        return report

    def get_currency_exposure_summary(self) -> pd.DataFrame:
        """
        Return audit table breaking down local asset currency, base currency,
        and currency drag / boost on annualized performance.
        """
        if self.asset_returns is None or self.local_asset_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        rows = []
        for asset_name, asset in self.config.assets.items():
            curr = asset.currency or self.base_currency
            ann_local = (
                calculate_annualized_return(self.local_asset_returns[asset_name], self.periods_per_year, geometric=True) * 100.0
                if asset_name in self.local_asset_returns else 0.0
            )
            ann_base = (
                calculate_annualized_return(self.asset_returns[asset_name], self.periods_per_year, geometric=True) * 100.0
                if asset_name in self.asset_returns else 0.0
            )
            fx_diff = ann_base - ann_local
            rows.append({
                "Asset": asset_name,
                "Quote Currency": curr,
                "Portfolio Base": self.base_currency,
                "Local Ann. Return (%)": ann_local,
                "Base Ann. Return (%)": ann_base,
                "FX Impact (%p)": fx_diff
            })
        return pd.DataFrame(rows).set_index("Asset")

    def get_performance_summary(self) -> pd.DataFrame:
        """Compute summary of basic return metrics across all assets and portfolio."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns

        summary_df = compute_return_summary(
            prices_or_returns=combined_returns,
            is_returns=True,
            periods_per_year=self.periods_per_year
        )
        return summary_df

    def get_comprehensive_risk_report(self, risk_free_rate: float = 0.0) -> pd.DataFrame:
        """Compute full risk, tail-risk, drawdown, and distribution analytics."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns

        return compute_comprehensive_risk_report(
            returns=combined_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=self.periods_per_year
        )

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation matrix of all assets and portfolio."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns
        return calculate_correlation_matrix(combined_returns)

    def get_covariance_matrix(self) -> pd.DataFrame:
        """Calculate annualized covariance matrix of all assets and portfolio."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns
        return calculate_covariance_matrix(combined_returns, periods_per_year=self.periods_per_year)

    def get_drawdown_series(self) -> pd.DataFrame:
        """Get percentage drawdown time series for all assets and portfolio."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns
        return calculate_drawdown_series(combined_returns, is_returns=True)

    def get_rolling_volatility(self, window: int = 60) -> pd.DataFrame:
        """Get rolling annualized volatility series."""
        if self.asset_returns is None or self.portfolio_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        combined_returns = self.asset_returns.copy()
        combined_returns["Portfolio"] = self.portfolio_returns
        return calculate_rolling_volatility(combined_returns, window=window, periods_per_year=self.periods_per_year)

    def get_rolling_correlation(self, asset_a: str, asset_b: str, window: int = 60) -> pd.Series:
        """Get rolling correlation series between two assets."""
        if self.asset_returns is None:
            raise RuntimeError("Portfolio market data is not loaded. Call load_data() first.")

        return calculate_rolling_correlation(
            series_a=self.asset_returns[asset_a],
            series_b=self.asset_returns[asset_b],
            window=window
        )

    def run_monte_carlo(self, config = None):
        """Execute Monte Carlo simulation using the portfolio's empirical parameters."""
        from ..simulation.monte_carlo import MonteCarloEngine, SimulationConfig
        cfg = config or SimulationConfig()
        engine = MonteCarloEngine()
        return engine.run_simulation(self, cfg)
