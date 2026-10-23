"""
Market Data Provider Abstraction.
Defines the base interface for market data ingestion and includes a deterministic
mock data provider for local testing and offline verification.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """
    Abstract Base Class for historical market data providers.
    Enables pluggable data backends (yfinance, Bloomberg, Polygon, CSV, etc.)
    without modifying core portfolio or analytics engines.
    """

    @abstractmethod
    def fetch_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices for a list of tickers.

        Args:
            tickers: List of asset ticker symbols (e.g. ['CSPX.L', 'BTC-USD']).
            start_date: ISO 8601 date string 'YYYY-MM-DD'.
            end_date: ISO 8601 date string 'YYYY-MM-DD'.
            frequency: Ingestion frequency ('daily', 'weekly', 'monthly').

        Returns:
            pd.DataFrame: DataFrame indexed by DatetimeIndex, columns are ticker symbols,
                          values are adjusted close prices (float).
        """
        pass

    @staticmethod
    def resample_prices(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Helper method to resample daily price data to weekly or monthly frequencies.

        Args:
            df: Daily price DataFrame with DatetimeIndex.
            frequency: 'daily', 'weekly', or 'monthly'.

        Returns:
            pd.DataFrame: Resampled price series using last available period close.
        """
        if df.empty:
            return df

        freq_lower = frequency.lower()
        if freq_lower in ["daily", "d"]:
            return df
        elif freq_lower in ["weekly", "w"]:
            # Resample weekly on Friday
            return df.resample("W-FRI").last().dropna(how="all")
        elif freq_lower in ["monthly", "m"]:
            # Handle pandas 2.2+ 'ME' vs older 'M'
            try:
                return df.resample("ME").last().dropna(how="all")
            except ValueError:
                return df.resample("M").last().dropna(how="all")
        else:
            raise ValueError(
                f"Unsupported frequency: '{frequency}'. Supported values are 'daily', 'weekly', 'monthly'."
            )


class MockMarketDataProvider(MarketDataProvider):
    """
    Deterministic Synthetic Market Data Provider.
    Generates realistic correlated or asset-specific geometric Brownian motion price paths
    for offline testing, reproducible unit tests, and CI/CD without network dependency.
    """

    DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
        "CSPX.L": {"initial_price": 350.0, "mu": 0.10, "sigma": 0.16},
        "CSPX": {"initial_price": 350.0, "mu": 0.10, "sigma": 0.16},
        "BTC-USD": {"initial_price": 20000.0, "mu": 0.35, "sigma": 0.65},
        "BBCA.JK": {"initial_price": 7500.0, "mu": 0.12, "sigma": 0.18},
        "BBRI.JK": {"initial_price": 4200.0, "mu": 0.11, "sigma": 0.22},
        "BMRI.JK": {"initial_price": 5500.0, "mu": 0.11, "sigma": 0.20},
        "GLD": {"initial_price": 160.0, "mu": 0.06, "sigma": 0.14},
        "GC=F": {"initial_price": 1800.0, "mu": 0.06, "sigma": 0.14},
        "USDIDR=X": {"initial_price": 15000.0, "mu": 0.02, "sigma": 0.08},
        "GBPUSD=X": {"initial_price": 1.25, "mu": 0.00, "sigma": 0.09},
        "EURUSD=X": {"initial_price": 1.08, "mu": 0.00, "sigma": 0.08},
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Generate synthetic price histories for requested tickers across business days.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(date_range) < 2:
            raise ValueError(f"Date range from {start_date} to {end_date} produces insufficient data points.")

        n_days = len(date_range)
        dt = 1.0 / 252.0
        rng = np.random.RandomState(self.seed)

        data = {}
        for ticker in tickers:
            params = self.DEFAULT_PARAMS.get(
                ticker,
                {"initial_price": 100.0, "mu": 0.08, "sigma": 0.20}
            )
            s0 = params["initial_price"]
            mu = params["mu"]
            sigma = params["sigma"]

            # Geometric Brownian Motion simulation
            shocks = rng.normal(
                loc=(mu - 0.5 * sigma**2) * dt,
                scale=sigma * np.sqrt(dt),
                size=n_days - 1
            )
            log_returns = np.concatenate(([0.0], shocks))
            price_path = s0 * np.exp(np.cumsum(log_returns))
            data[ticker] = price_path

        df = pd.DataFrame(data, index=date_range)
        df.index.name = "Date"

        return self.resample_prices(df, frequency)
