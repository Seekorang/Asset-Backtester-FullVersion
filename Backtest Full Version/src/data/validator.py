"""
Market Data Validation & Quality Engine.
Performs missing data detection, duplicate detection, date alignment across different
market calendars (Crypto 24/7 vs Equities/Commodities), timezone normalization,
outlier detection, and produces comprehensive auditable Data Quality Reports without lookahead bias.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """
    Structured report auditing the quality and integrity of ingested market data.
    """
    start_date: str
    end_date: str
    total_timestamps: int
    tickers: List[str]
    missing_counts: Dict[str, int] = field(default_factory=dict)
    missing_percentages: Dict[str, float] = field(default_factory=dict)
    imputed_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    non_positive_counts: Dict[str, int] = field(default_factory=dict)
    trading_days_per_ticker: Dict[str, int] = field(default_factory=dict)
    outliers_detected: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    alignment_method: str = "forward_fill"
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True

    def summary_text(self) -> str:
        """Format the report into a clean, human-readable audit summary."""
        lines = [
            "================ DATA QUALITY AUDIT REPORT ================",
            f"Period Covered: {self.start_date} to {self.end_date} ({self.total_timestamps} periods)",
            f"Assets Audited: {', '.join(self.tickers)}",
            f"Alignment Strategy: {self.alignment_method}",
            f"Duplicates Detected: {self.duplicate_count}",
            "----------------- Ticker Trading & Imputation -----------------",
        ]
        for ticker in self.tickers:
            active_days = self.trading_days_per_ticker.get(ticker, 0)
            n_missing = self.missing_counts.get(ticker, 0)
            pct_missing = self.missing_percentages.get(ticker, 0.0)
            imputed = self.imputed_counts.get(ticker, 0)
            non_pos = self.non_positive_counts.get(ticker, 0)
            lines.append(
                f"* {ticker:<12} | Active: {active_days:>5} | Missing: {n_missing:>4} ({pct_missing:>5.2f}%) | Imputed: {imputed:>4} | Non-positive: {non_pos}"
            )

        total_outliers = sum(len(v) for v in self.outliers_detected.values())
        lines.append(f"Outliers Detected (Statistical Anomalies): {total_outliers}")
        if total_outliers > 0:
            for ticker, items in self.outliers_detected.items():
                if items:
                    sample = items[:3]
                    sample_str = ", ".join(f"{it['date']}: {it['return_pct']:+.1f}%" for it in sample)
                    if len(items) > 3:
                        sample_str += f" (+{len(items)-3} more)"
                    lines.append(f"  - {ticker}: {len(items)} outliers ({sample_str})")

        if self.warnings:
            lines.append("------------------------- Warnings -------------------------")
            for w in self.warnings:
                lines.append(f"[!] {w}")
        else:
            lines.append("Status: All data passed integrity and validation checks.")

        lines.append("============================================================")
        return "\n".join(lines)


class DataValidator:
    """
    Validates, sanitizes, and aligns multi-asset price time series without lookahead bias.
    """

    def __init__(
        self,
        outlier_std_threshold: float = 4.0,
        max_allowed_missing_pct: float = 0.35,
    ):
        self.outlier_std_threshold = outlier_std_threshold
        self.max_allowed_missing_pct = max_allowed_missing_pct

    def validate_and_align(
        self,
        df: pd.DataFrame,
        alignment_strategy: str = "forward_fill"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Perform complete validation and multi-asset calendar alignment.

        Args:
            df: Raw price DataFrame with DatetimeIndex and ticker columns.
            alignment_strategy: 'forward_fill', 'business_days', or 'intersection'.

        Returns:
            Tuple[pd.DataFrame, DataQualityReport]: Cleaned aligned DataFrame and the audit report.
        """
        if df.empty:
            raise ValueError("Supplied DataFrame is empty. Cannot perform data validation.")

        warnings: List[str] = []
        cleaned_df = df.copy()

        # 1. Timezone Normalization
        if cleaned_df.index.tz is not None:
            cleaned_df.index = cleaned_df.index.tz_convert(None)
        cleaned_df.index = pd.to_datetime(cleaned_df.index)
        cleaned_df.index.name = "Date"

        # 2. Duplicate Detection & Deduplication
        dup_count = int(cleaned_df.index.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate timestamp(s) in index. Keeping the last record per timestamp.")
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="last")]

        # 3. Sort chronologically using stable sort
        cleaned_df = cleaned_df.sort_index(kind="stable")

        tickers = list(cleaned_df.columns)
        trading_days: Dict[str, int] = {}
        missing_counts: Dict[str, int] = {}
        missing_pcts: Dict[str, float] = {}
        imputed_counts: Dict[str, int] = {}
        non_positive: Dict[str, int] = {}
        outliers: Dict[str, List[Dict[str, Any]]] = {}

        total_raw_rows = len(cleaned_df)

        # 4. Check initial per-asset metrics & non-positive values
        raw_null_masks = {}
        for col in tickers:
            series = cleaned_df[col]
            valid_mask = series.notna()
            raw_null_masks[col] = ~valid_mask
            trading_days[col] = int(valid_mask.sum())
            n_missing = int(series.isna().sum())
            missing_counts[col] = n_missing
            pct_missing = (n_missing / total_raw_rows) * 100.0 if total_raw_rows > 0 else 0.0
            missing_pcts[col] = pct_missing

            invalid_prices = int((series[valid_mask] <= 0).sum())
            non_positive[col] = invalid_prices
            if invalid_prices > 0:
                warnings.append(f"Ticker '{col}' contains {invalid_prices} non-positive (<=0) price records.")

            if pct_missing > (self.max_allowed_missing_pct * 100.0):
                warnings.append(
                    f"Ticker '{col}' has {pct_missing:.1f}% missing data prior to alignment, exceeding threshold ({self.max_allowed_missing_pct*100:.0f}%)."
                )

        # 5. Asset Trading Calendar Alignment (Zero Lookahead Bias)
        strategy_lower = alignment_strategy.lower()
        if strategy_lower == "forward_fill":
            # Forward-fill past observations into missing dates
            ffilled = cleaned_df.ffill()
            # Trim leading NaNs before assets started trading (strictly NO bfill to prevent lookahead bias)
            valid_start = ffilled.dropna(how="any").index
            if len(valid_start) > 0:
                cleaned_df = ffilled.loc[valid_start[0]:]
            else:
                cleaned_df = ffilled.dropna(how="all")
        elif strategy_lower in ["business_days", "bdate"]:
            # Institutional multi-asset convention: reindex to business days, avoiding weekend crypto zeros
            b_index = pd.bdate_range(start=cleaned_df.index[0], end=cleaned_df.index[-1])
            reindexed = cleaned_df.reindex(b_index).ffill()
            valid_start = reindexed.dropna(how="any").index
            if len(valid_start) > 0:
                cleaned_df = reindexed.loc[valid_start[0]:]
            else:
                cleaned_df = reindexed.dropna(how="all")
            cleaned_df.index.name = "Date"
        elif strategy_lower == "intersection":
            cleaned_df = cleaned_df.dropna(how="any")
        else:
            raise ValueError(
                f"Unknown alignment strategy: '{alignment_strategy}'. Choose 'forward_fill', 'business_days', or 'intersection'."
            )

        if cleaned_df.empty:
            raise ValueError("All data rows were eliminated during calendar alignment and NaN handling.")

        # Compute imputed count (values filled in where raw data was missing)
        for col in tickers:
            if col in cleaned_df.columns:
                orig_col = df[col] if col in df.columns else None
                if orig_col is not None:
                    # In cleaned_df, count records where raw input was null or not indexed
                    deduped_orig = orig_col[~orig_col.index.duplicated(keep="last")]
                    raw_reindexed = deduped_orig.reindex(cleaned_df.index)
                    imputed_count = int(raw_reindexed.isna().sum())
                    imputed_counts[col] = imputed_count
                else:
                    imputed_counts[col] = 0

        # 6. Outlier Detection on Daily Returns
        daily_rets = cleaned_df.pct_change().dropna(how="all")
        for col in tickers:
            rets = daily_rets[col].dropna()
            if len(rets) > 20:
                mean_r = rets.mean()
                std_r = rets.std()
                if std_r > 0:
                    z_scores = (rets - mean_r).abs() / std_r
                    outlier_mask = (z_scores > self.outlier_std_threshold) | (rets.abs() > 0.50)
                    outlier_records = []
                    for dt, ret_val in rets[outlier_mask].items():
                        outlier_records.append({
                            "date": dt.strftime("%Y-%m-%d"),
                            "return_pct": float(ret_val * 100.0),
                            "z_score": float((abs(ret_val - mean_r)) / std_r)
                        })
                    outliers[col] = outlier_records
                    if len(outlier_records) > 0:
                        warnings.append(
                            f"Ticker '{col}' has {len(outlier_records)} daily return outlier(s) (|z| > {self.outlier_std_threshold} or |ret| > 50%)."
                        )
            else:
                outliers[col] = []

        start_str = cleaned_df.index[0].strftime("%Y-%m-%d")
        end_str = cleaned_df.index[-1].strftime("%Y-%m-%d")

        report = DataQualityReport(
            start_date=start_str,
            end_date=end_str,
            total_timestamps=len(cleaned_df),
            tickers=tickers,
            missing_counts=missing_counts,
            missing_percentages=missing_pcts,
            imputed_counts=imputed_counts,
            duplicate_count=dup_count,
            non_positive_counts=non_positive,
            trading_days_per_ticker=trading_days,
            outliers_detected=outliers,
            alignment_method=alignment_strategy,
            warnings=warnings,
            is_valid=(sum(non_positive.values()) == 0 and len(cleaned_df) >= 10)
        )

        return cleaned_df, report
