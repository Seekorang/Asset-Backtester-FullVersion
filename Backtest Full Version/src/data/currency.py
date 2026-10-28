"""
Multi-Currency & FX Alignment Engine.
Handles currency identification, quote conventions, exchange rate alignment,
and multiplicative return conversion: (1 + R_base) = (1 + R_local) * (1 + R_fx).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Standard default currency mappings for common asset suffixes/patterns
DEFAULT_TICKER_CURRENCIES: Dict[str, str] = {
    "CSPX.L": "USD",   # iShares Core S&P 500 UCITS ETF trades on LSE in USD
    "CSP1.L": "GBP",   # GBP equivalent
    "BTC-USD": "USD",  # Bitcoin quoted in USD
    "BBCA.JK": "IDR",  # Indonesia Stock Exchange listed in IDR
    "BBRI.JK": "IDR",
    "BMRI.JK": "IDR",
    "BBNI.JK": "IDR",
    "GLD": "USD",      # SPDR Gold Shares NYSE Arca in USD
    "GC=F": "USD",     # Gold Futures in USD
    "VOO": "USD",      # Vanguard S&P 500 in USD
    "SPY": "USD",      # SPDR S&P 500 in USD
}


def infer_currency_from_ticker(ticker: str) -> str:
    """Infer quote currency from ticker suffix or lookup table."""
    clean = ticker.strip().upper()
    if clean in DEFAULT_TICKER_CURRENCIES:
        return DEFAULT_TICKER_CURRENCIES[clean]
    if clean.endswith(".JK"):
        return "IDR"
    if clean.endswith(".L"):
        return "USD"  # Default LSE international ETF quote
    if clean.endswith("-USD"):
        return "USD"
    if clean.endswith(".TO"):
        return "CAD"
    if clean.endswith(".AX"):
        return "AUD"
    return "USD"


def get_required_fx_tickers(
    asset_currencies: Dict[str, str],
    base_currency: str = "USD"
) -> List[str]:
    """
    Determine which FX rate time-series need to be retrieved from market data provider.
    Currently supports cross-conversions between USD, IDR, and GBP.
    """
    base_curr = base_currency.strip().upper()
    required_fx = set()

    for ticker, curr in asset_currencies.items():
        c_upper = curr.strip().upper()
        if c_upper == base_curr:
            continue
        # Standard Yahoo Finance FX ticker pairs
        if (base_curr == "USD" and c_upper == "IDR") or (base_curr == "IDR" and c_upper == "USD"):
            required_fx.add("USDIDR=X")
        elif (base_curr == "USD" and c_upper == "GBP") or (base_curr == "GBP" and c_upper == "USD"):
            required_fx.add("GBPUSD=X")
        elif (base_curr == "USD" and c_upper == "EUR") or (base_curr == "EUR" and c_upper == "USD"):
            required_fx.add("EURUSD=X")
        else:
            # General fallback pair
            required_fx.add(f"{c_upper}{base_curr}=X")

    return sorted(list(required_fx))


def convert_asset_prices_to_base(
    prices: pd.Series,
    from_currency: str,
    base_currency: str,
    fx_rates_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Convert a price time series from local currency to portfolio base currency.

    Quotation Conventions:
    - USDIDR=X: 1 USD = S IDR (e.g. 15,500 IDR).
      To convert IDR -> USD: P_USD = P_IDR / S
      To convert USD -> IDR: P_IDR = P_USD * S
    - GBPUSD=X: 1 GBP = S USD (e.g. 1.28 USD).
      To convert GBP -> USD: P_USD = P_GBP * S
      To convert USD -> GBP: P_GBP = P_USD / S
    """
    from_curr = from_currency.strip().upper()
    to_curr = base_currency.strip().upper()

    if from_curr == to_curr:
        return prices.copy()

    if fx_rates_df is None or fx_rates_df.empty:
        raise ValueError(
            f"FX rates required to convert from {from_curr} to {to_curr}, but no FX data provided."
        )

    # 1. USD <-> IDR
    if (from_curr == "IDR" and to_curr == "USD"):
        if "USDIDR=X" not in fx_rates_df.columns:
            raise KeyError("USDIDR=X FX rate series missing from market data.")
        fx_aligned = fx_rates_df["USDIDR=X"].reindex(prices.index).ffill().bfill()
        return prices / fx_aligned

    if (from_curr == "USD" and to_curr == "IDR"):
        if "USDIDR=X" not in fx_rates_df.columns:
            raise KeyError("USDIDR=X FX rate series missing from market data.")
        fx_aligned = fx_rates_df["USDIDR=X"].reindex(prices.index).ffill().bfill()
        return prices * fx_aligned

    # 2. GBP <-> USD
    if (from_curr == "GBP" and to_curr == "USD"):
        if "GBPUSD=X" not in fx_rates_df.columns:
            raise KeyError("GBPUSD=X FX rate series missing from market data.")
        fx_aligned = fx_rates_df["GBPUSD=X"].reindex(prices.index).ffill().bfill()
        return prices * fx_aligned

    if (from_curr == "USD" and to_curr == "GBP"):
        if "GBPUSD=X" not in fx_rates_df.columns:
            raise KeyError("GBPUSD=X FX rate series missing from market data.")
        fx_aligned = fx_rates_df["GBPUSD=X"].reindex(prices.index).ffill().bfill()
        return prices / fx_aligned

    raise NotImplementedError(f"Currency conversion from {from_curr} to {to_curr} is not yet implemented.")


def convert_asset_returns_to_base(
    local_returns: pd.Series,
    from_currency: str,
    base_currency: str,
    fx_rates_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Convert simple periodic returns from local currency into base currency returns
    using exact multiplicative compounding:
    (1 + R_base) = (1 + R_local) * (1 + R_fx)
    R_base = (1 + R_local) * (1 + R_fx) - 1
    """
    from_curr = from_currency.strip().upper()
    to_curr = base_currency.strip().upper()

    if from_curr == to_curr:
        return local_returns.copy()

    if fx_rates_df is None or fx_rates_df.empty:
        raise ValueError(
            f"FX rates required to convert returns from {from_curr} to {to_curr}, but no FX data provided."
        )

    # Calculate exact FX return multiplier
    # Case A: IDR to USD with S = USDIDR=X (IDR per USD)
    # V_USD_t = P_IDR_t / S_t
    # 1 + R_USD = (P_t / P_{t-1}) * (S_{t-1} / S_t)
    if from_curr == "IDR" and to_curr == "USD":
        if "USDIDR=X" not in fx_rates_df.columns:
            raise KeyError("USDIDR=X FX rate series missing from market data.")
        s_full = fx_rates_df["USDIDR=X"].ffill()
        fx_mult_full = s_full.shift(1) / s_full
        fx_multiplier = fx_mult_full.reindex(local_returns.index).fillna(1.0)
        base_returns = (1.0 + local_returns) * fx_multiplier - 1.0
        return base_returns

    # Case B: USD to IDR with S = USDIDR=X (IDR per USD)
    # V_IDR_t = P_USD_t * S_t
    # 1 + R_IDR = (P_t / P_{t-1}) * (S_t / S_{t-1})
    if from_curr == "USD" and to_curr == "IDR":
        if "USDIDR=X" not in fx_rates_df.columns:
            raise KeyError("USDIDR=X FX rate series missing from market data.")
        s_full = fx_rates_df["USDIDR=X"].ffill()
        fx_mult_full = s_full / s_full.shift(1)
        fx_multiplier = fx_mult_full.reindex(local_returns.index).fillna(1.0)
        base_returns = (1.0 + local_returns) * fx_multiplier - 1.0
        return base_returns

    # Case C: GBP to USD with S = GBPUSD=X (USD per GBP)
    # V_USD_t = P_GBP_t * S_t
    if from_curr == "GBP" and to_curr == "USD":
        if "GBPUSD=X" not in fx_rates_df.columns:
            raise KeyError("GBPUSD=X FX rate series missing from market data.")
        s_full = fx_rates_df["GBPUSD=X"].ffill()
        fx_mult_full = s_full / s_full.shift(1)
        fx_multiplier = fx_mult_full.reindex(local_returns.index).fillna(1.0)
        base_returns = (1.0 + local_returns) * fx_multiplier - 1.0
        return base_returns

    raise NotImplementedError(f"Return conversion from {from_curr} to {to_curr} is not yet implemented.")
