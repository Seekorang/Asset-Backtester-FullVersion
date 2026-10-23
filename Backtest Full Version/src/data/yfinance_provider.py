"""
Yahoo Finance Data Provider Implementation.
Fetches historical market data using yfinance, extracts Adjusted Close prices,
normalizes timezones, and provides resilient multi-strategy downloading
with individual ticker fallbacks and Net Total Return dividend tax adjustments.
"""

from typing import List, Optional, Dict
import datetime
import time
import logging
import pandas as pd
from .provider import MarketDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """
    MarketDataProvider implementation using Yahoo Finance (via yfinance package).
    Features robust multi-strategy fetching: batch download with automatic
    individual ticker fallback, retries, and Net Total Return dividend tax adjustments.
    """

    def __init__(
        self,
        auto_adjust: bool = False,
        dividend_tax_rates: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            auto_adjust: If True, yfinance auto-adjusts OHLC. If False, we explicitly
                         target 'Adj Close' to account for dividends and splits.
            dividend_tax_rates: Optional dictionary mapping ticker symbols to dividend tax rates
                                (e.g. {'BBCA.JK': 0.10} for 10% Indonesian PPh Final dividend tax).
                                Converts Gross Total Return (Adj Close) into Net Total Return.
        """
        self.auto_adjust = auto_adjust
        # Default: 10% dividend withholding tax on Indonesian equity (BBCA.JK) under UU HPP
        self.dividend_tax_rates: Dict[str, float] = (
            dividend_tax_rates if dividend_tax_rates is not None else {"BBCA.JK": 0.10}
        )

    def fetch_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Download historical price data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols (e.g. ['CSPX.L', 'BTC-USD', 'BBCA.JK', 'GLD']).
            start_date: Start date string 'YYYY-MM-DD'.
            end_date: End date string 'YYYY-MM-DD', 'today', or None.
            frequency: 'daily', 'weekly', or 'monthly'.

        Returns:
            pd.DataFrame: Aligned adjusted close prices indexed by Date.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is not installed in the environment. "
                "Please run: pip install yfinance"
            ) from e

        if not tickers:
            return pd.DataFrame()

        clean_tickers = [t.strip() for t in tickers if t.strip()]

        # 1. Resolve dynamic end_date ('today', 'now', None, or future dates)
        today_date = datetime.date.today()
        today_str = today_date.strftime("%Y-%m-%d")

        s_date = str(start_date).strip()
        e_str = str(end_date).strip() if end_date else today_str

        if e_str.lower() in ("today", "now", "none", ""):
            resolved_end_date = today_str
        else:
            try:
                parsed_e = datetime.date.fromisoformat(e_str)
                if parsed_e > today_date:
                    logger.info(f"Clamping future end_date '{e_str}' to today: {today_str}")
                    resolved_end_date = today_str
                else:
                    resolved_end_date = e_str
            except Exception:
                resolved_end_date = today_str

        logger.info(f"Fetching market data for {clean_tickers} from {s_date} to {resolved_end_date} via yfinance...")

        price_dict: Dict[str, pd.Series] = {}

        # 2. Strategy A: Batch Download via yf.download (single network call)
        try:
            raw_data = yf.download(
                tickers=clean_tickers,
                start=s_date,
                end=resolved_end_date,
                auto_adjust=self.auto_adjust,
                progress=False,
                threads=False
            )
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    level_0 = raw_data.columns.get_level_values(0)
                    target_field = (
                        "Adj Close" if "Adj Close" in level_0 and not self.auto_adjust
                        else ("Close" if "Close" in level_0 else None)
                    )
                    if target_field:
                        sub_df = raw_data[target_field]
                        for t in clean_tickers:
                            if t in sub_df.columns:
                                s = sub_df[t].dropna()
                                if not s.empty:
                                    price_dict[t] = s
                else:
                    target_col = (
                        "Adj Close" if "Adj Close" in raw_data.columns and not self.auto_adjust
                        else ("Close" if "Close" in raw_data.columns else None)
                    )
                    if target_col and len(clean_tickers) == 1:
                        s = raw_data[target_col].dropna()
                        if not s.empty:
                            price_dict[clean_tickers[0]] = s
        except Exception as exc:
            logger.warning(f"Batch yfinance download encountered warning: {exc}. Will use individual ticker fetcher.")

        # 3. Strategy B: Fallback for any missing or failed tickers (ticker-by-ticker with retry)
        missing_tickers = [t for t in clean_tickers if t not in price_dict or price_dict[t].empty]
        if missing_tickers:
            logger.info(f"Downloading {len(missing_tickers)} ticker(s) individually via yf.Ticker fallback: {missing_tickers}...")
            for ticker in missing_tickers:
                success = False
                for attempt in range(1, 4):
                    try:
                        tk = yf.Ticker(ticker)
                        hist = tk.history(start=s_date, end=resolved_end_date, auto_adjust=self.auto_adjust)
                        if not hist.empty:
                            target_col = "Adj Close" if ("Adj Close" in hist.columns and not self.auto_adjust) else "Close"
                            if target_col in hist.columns:
                                s = hist[target_col].dropna()
                                if not s.empty:
                                    price_dict[ticker] = s
                                    success = True
                                    break

                        # Alternative single-ticker download
                        single_raw = yf.download(
                            ticker,
                            start=s_date,
                            end=resolved_end_date,
                            auto_adjust=self.auto_adjust,
                            progress=False
                        )
                        if not single_raw.empty:
                            target_col = "Adj Close" if ("Adj Close" in single_raw.columns and not self.auto_adjust) else "Close"
                            if target_col in single_raw.columns:
                                s = single_raw[target_col].dropna()
                                if not s.empty:
                                    price_dict[ticker] = s
                                    success = True
                                    break
                    except Exception as err:
                        logger.warning(f"Attempt {attempt}/3 failed for ticker '{ticker}': {err}")
                        time.sleep(0.5)

                if not success:
                    logger.error(f"Failed to retrieve price data for ticker '{ticker}' across all fetch strategies.")

        if not price_dict:
            raise ValueError(
                f"No price data could be retrieved by yfinance for tickers: {clean_tickers} "
                f"between {s_date} and {resolved_end_date}. Please check internet connection or Yahoo Finance status."
            )

        # 4. Net-of-Tax Total Return Adjustment (e.g. for BBCA.JK 10% PPh Final Dividend Tax)
        for ticker in clean_tickers:
            if ticker in price_dict and ticker in self.dividend_tax_rates:
                tax_rate = self.dividend_tax_rates[ticker]
                if tax_rate > 0.0:
                    try:
                        tk = yf.Ticker(ticker)
                        divs = tk.dividends
                        if not divs.empty:
                            divs_df = divs.copy()
                            divs_df.index = pd.to_datetime(divs_df.index)
                            if divs_df.index.tz is not None:
                                divs_df.index = divs_df.index.tz_localize(None)
                            divs_df.index = divs_df.index.normalize()
                            divs_df = divs_df[~divs_df.index.duplicated(keep="last")]

                            raw_s = price_dict[ticker].copy()
                            raw_s.index = pd.to_datetime(raw_s.index)
                            if raw_s.index.tz is not None:
                                raw_s.index = raw_s.index.tz_localize(None)
                            raw_s.index = raw_s.index.normalize()

                            aligned_divs = divs_df.reindex(raw_s.index).fillna(0.0)
                            if (aligned_divs > 0).any():
                                p_prev = raw_s.shift(1)
                                div_yield = (aligned_divs / p_prev).fillna(0.0)
                                gross_rets = raw_s.pct_change().fillna(0.0)
                                # Subtract dividend withholding tax from gross return on ex-dividend dates
                                net_rets = gross_rets - (tax_rate * div_yield)
                                net_prices = (1.0 + net_rets).cumprod() * raw_s.iloc[0]
                                price_dict[ticker] = net_prices
                                logger.info(
                                    f"Applied Net Total Return adjustment to '{ticker}' "
                                    f"(deducted {tax_rate*100:.1f}% dividend tax across {(aligned_divs > 0).sum()} dividend events)."
                                )
                    except Exception as err:
                        logger.warning(f"Could not apply net dividend adjustment to '{ticker}': {err}. Using gross Adj Close.")

        # 5. Standardize, align index and deduplicate
        price_df = pd.DataFrame(price_dict)
        price_df.index = pd.to_datetime(price_df.index)
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)
        price_df.index = price_df.index.normalize()
        price_df = price_df.groupby(price_df.index).last()
        price_df.index.name = "Date"

        available_cols = [t for t in clean_tickers if t in price_df.columns]
        price_df = price_df[available_cols]

        return self.resample_prices(price_df, frequency)
