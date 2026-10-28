"""
Yahoo Finance Data Provider Implementation.
Fetches historical market data using yfinance, extracts Adjusted Close prices,
normalizes timezones, and provides resilient multi-strategy downloading
with individual ticker fallbacks and Net Total Return dividend tax adjustments.
"""

from typing import List, Optional, Dict
from pathlib import Path
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
    individual ticker fallback, persistent local caching, retries, and Net Total
    Return dividend tax adjustments.
    """

    @staticmethod
    def _get_cache_path(ticker: str) -> Path:
        cache_dir = Path(".cache") / "market_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
        return cache_dir / f"{safe_ticker}.csv"

    @classmethod
    def _save_to_cache(cls, ticker: str, series: pd.Series) -> None:
        try:
            path = cls._get_cache_path(ticker)
            df = series.to_frame(name="Price")
            df.index.name = "Date"
            df.to_csv(path)
        except Exception as exc:
            logger.debug(f"Could not save cache for {ticker}: {exc}")

    @classmethod
    def _load_from_cache(cls, ticker: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        try:
            path = cls._get_cache_path(ticker)
            if not path.exists():
                return None
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if df.empty or "Price" not in df.columns:
                return None
            s = df["Price"].dropna()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if hasattr(s, "squeeze"):
                s = s.squeeze()
            if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = s.index.normalize()
            s = s.sort_index()
            sub_s = s.loc[start_date:end_date]
            if len(sub_s) >= 10:
                return sub_s
            elif len(s) >= 10:
                return s
        except Exception as exc:
            logger.debug(f"Could not load cache for {ticker}: {exc}")
        return None

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
        # Find last fully closed global trading day (yesterday or previous Friday)
        last_complete_date = today_date - datetime.timedelta(days=1)
        while last_complete_date.weekday() >= 5:
            last_complete_date -= datetime.timedelta(days=1)
        last_complete_str = last_complete_date.strftime("%Y-%m-%d")

        s_date = str(start_date).strip()
        e_str = str(end_date).strip() if end_date else last_complete_str

        if e_str.lower() in ("today", "now", "none", ""):
            resolved_end_date = last_complete_str
            api_end_date = None
        else:
            try:
                parsed_e = datetime.date.fromisoformat(e_str)
                if parsed_e >= today_date:
                    logger.info(f"Clamping unclosed/today end_date '{e_str}' to last complete market close: {last_complete_str}")
                    resolved_end_date = last_complete_str
                    api_end_date = None
                else:
                    resolved_end_date = e_str
                    api_end_date = None
            except Exception:
                resolved_end_date = last_complete_str
                api_end_date = None

        logger.info(f"Fetching market data for {clean_tickers} from {s_date} to {resolved_end_date} via yfinance...")

        price_dict: Dict[str, pd.Series] = {}

        # 2. Strategy A: Batch Download via yf.download (single network call)
        try:
            raw_data = yf.download(
                tickers=clean_tickers,
                start=s_date,
                end=api_end_date,
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
                                if not s.empty and len(s) >= 5:
                                    price_dict[t] = s
                else:
                    target_col = (
                        "Adj Close" if "Adj Close" in raw_data.columns and not self.auto_adjust
                        else ("Close" if "Close" in raw_data.columns else None)
                    )
                    if target_col and len(clean_tickers) == 1:
                        s = raw_data[target_col].dropna()
                        if not s.empty and len(s) >= 5:
                            price_dict[clean_tickers[0]] = s
        except Exception as exc:
            logger.warning(f"Batch yfinance download encountered warning: {exc}. Will use individual ticker fetcher.")

        # 3. Strategy B: Resilient Fallback for missing tickers with backoff and cache
        missing_tickers = [t for t in clean_tickers if t not in price_dict or price_dict[t].empty or len(price_dict[t]) < 5]
        if missing_tickers:
            logger.info(f"Downloading {len(missing_tickers)} ticker(s) individually via resilient fallback: {missing_tickers}...")
            for ticker in missing_tickers:
                success = False
                for attempt in range(1, 4):
                    try:
                        tk = yf.Ticker(ticker)
                        # Attempt 1: tk.history with api_end_date
                        hist = tk.history(start=s_date, end=api_end_date, auto_adjust=self.auto_adjust)
                        if not hist.empty:
                            target_col = "Adj Close" if ("Adj Close" in hist.columns and not self.auto_adjust) else "Close"
                            if target_col in hist.columns:
                                s = hist[target_col].dropna()
                                if len(s) >= 5:
                                    price_dict[ticker] = s
                                    success = True
                                    break

                        # Attempt 2: yf.download single ticker with end=None
                        single_raw = yf.download(
                            ticker,
                            start=s_date,
                            end=None,
                            auto_adjust=self.auto_adjust,
                            progress=False
                        )
                        if not single_raw.empty:
                            target_col = "Adj Close" if ("Adj Close" in single_raw.columns and not self.auto_adjust) else "Close"
                            if target_col in single_raw.columns:
                                s = single_raw[target_col].dropna()
                                if len(s) >= 5:
                                    price_dict[ticker] = s
                                    success = True
                                    break

                        # Attempt 3: tk.history period='max' filtered by start_date
                        max_hist = tk.history(period="max", auto_adjust=self.auto_adjust)
                        if not max_hist.empty:
                            target_col = "Adj Close" if ("Adj Close" in max_hist.columns and not self.auto_adjust) else "Close"
                            if target_col in max_hist.columns:
                                s = max_hist[target_col].dropna()
                                if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                                    s.index = s.index.tz_localize(None)
                                s = s.loc[s_date:]
                                if len(s) >= 5:
                                    price_dict[ticker] = s
                                    success = True
                                    break
                    except Exception as err:
                        logger.warning(f"Attempt {attempt}/3 failed for ticker '{ticker}': {err}")

                    if not success:
                        time.sleep(1.0 * attempt)

                if not success:
                    # Strategy C: Offline fallback to local persistent cache
                    cached_s = self._load_from_cache(ticker, s_date, resolved_end_date)
                    if cached_s is not None and not cached_s.empty:
                        price_dict[ticker] = cached_s
                        success = True
                        logger.warning(
                            f"Remote fetch failed for '{ticker}'. Successfully loaded {len(cached_s)} "
                            f"historical prices from local persistent cache."
                        )
                    else:
                        logger.error(f"Failed to retrieve price data for ticker '{ticker}' across all online and cached strategies.")

        if not price_dict:
            raise ValueError(
                f"No price data could be retrieved by yfinance for tickers: {clean_tickers} "
                f"between {s_date} and {resolved_end_date}. Please check internet connection or Yahoo Finance status."
            )

        # Save successful prices to persistent cache for offline resilience (up to resolved_end_date)
        for ticker in clean_tickers:
            if ticker in price_dict and not price_dict[ticker].empty:
                s_to_cache = price_dict[ticker]
                if isinstance(s_to_cache.index, pd.DatetimeIndex):
                    s_to_cache = s_to_cache.loc[:resolved_end_date]
                self._save_to_cache(ticker, s_to_cache)

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
        # Ensure every individual series has tz-naive normalized DatetimeIndex before DataFrame union
        for t in list(price_dict.keys()):
            s = price_dict[t]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if hasattr(s, "squeeze"):
                s = s.squeeze()
            s_idx = pd.to_datetime(s.index)
            if s_idx.tz is not None:
                s_idx = s_idx.tz_localize(None)
            s_idx = s_idx.normalize()
            s.index = s_idx
            price_dict[t] = s.groupby(s.index).last()

        price_df = pd.DataFrame(price_dict)
        price_df.index = pd.to_datetime(price_df.index)
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)
        price_df.index = price_df.index.normalize()
        price_df = price_df.groupby(price_df.index).last()
        price_df.index.name = "Date"

        available_cols = [t for t in clean_tickers if t in price_df.columns]
        price_df = price_df[available_cols]

        # Guarantee no future dates beyond resolved_end_date leak into analysis
        price_df = price_df.loc[:resolved_end_date]

        return self.resample_prices(price_df, frequency)
