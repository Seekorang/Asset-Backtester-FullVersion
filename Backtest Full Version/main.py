"""
Personal Portfolio Wealth Simulator & Tax-Alpha Research Engine
Integrated Quantitative Execution Script (Multi-Currency & USD Base Denominated).
"""

import sys
import os
import argparse
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# Safe unicode encoding setup for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.portfolio.allocation import PortfolioConfig
from src.portfolio.portfolio import Portfolio
from src.portfolio.optimizer import PortfolioOptimizer
from src.data.provider import MockMarketDataProvider
from src.data.yfinance_provider import YFinanceProvider
from src.analytics.correlation import format_correlation_heatmap_text, calculate_regime_switching_correlation
from src.analytics.validation import WalkForwardValidator, StressTestEngine
from src.analytics.benchmark import compute_benchmark_report, resample_returns_weekly
from src.analytics.returns import calculate_annualized_return as calculate_annualized_return_fn
from src.backtest.backtester import Backtester, BacktestConfig
from src.simulation.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationModel
from src.tax.tax_engine import TaxEngine
from src.tax.tax_alpha import TaxAlphaEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PortfolioEngine")


def fetch_irx_yield_averages(
    provider,
    target_provider: str,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None
) -> Tuple[float, float]:
    """
    Dynamically fetch 3-Month Treasury Bill Yield (^IRX) to compute historical risk-free rate
    benchmarks (2016-2021 bull period average vs full period average).
    Features persistent disk caching and period='max' to eliminate CBOE boundary errors.
    """
    irx_mean_bull, irx_mean_full = 0.93, 2.27
    if target_provider.lower() == "yfinance":
        try:
            import yfinance as yf
            irx_s = None
            # Check local cache first
            cached_irx = YFinanceProvider._load_from_cache("^IRX", start_date, "2099-12-31")
            if cached_irx is not None and len(cached_irx) >= 10:
                irx_s = cached_irx

            # Try online fetch using period='max' (immune to CBOE intraday/end boundary errors)
            if irx_s is None or len(irx_s) < 10:
                for attempt in range(1, 3):
                    try:
                        tk_irx = yf.Ticker("^IRX")
                        hist_irx = tk_irx.history(period="max", auto_adjust=False)
                        if not hist_irx.empty:
                            col = "Adj Close" if "Adj Close" in hist_irx.columns else "Close"
                            s = hist_irx[col].dropna()
                            if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                                s.index = s.index.tz_localize(None)
                            s = s.loc[start_date:]
                            if len(s) >= 10:
                                irx_s = s
                                YFinanceProvider._save_to_cache("^IRX", irx_s)
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)

            if (irx_s is None or len(irx_s) < 10) and cached_irx is not None:
                irx_s = cached_irx

            if irx_s is not None and not irx_s.empty:
                if isinstance(irx_s.index, pd.DatetimeIndex) and irx_s.index.tz is not None:
                    irx_s.index = irx_s.index.tz_localize(None)
                sub_bull = irx_s.loc["2016-01-01":"2021-12-31"]
                if len(sub_bull) > 0:
                    irx_mean_bull = float(sub_bull.mean())
                if len(irx_s) > 0:
                    irx_mean_full = float(irx_s.mean())
        except Exception:
            pass
    else:
        try:
            irx_mock = provider.fetch_history(["^IRX"], start_date=start_date, end_date=end_date)
            if not irx_mock.empty and "^IRX" in irx_mock.columns:
                irx_s = irx_mock["^IRX"].dropna()
                sub_bull = irx_s.loc["2016-01-01":"2021-12-31"]
                if len(sub_bull) > 0:
                    irx_mean_bull = float(sub_bull.mean())
                if len(irx_s) > 0:
                    irx_mean_full = float(irx_s.mean())
        except Exception:
            pass
    return float(irx_mean_bull), float(irx_mean_full)


def run_pipeline(
    config_path: str = None,
    tax_config_path: str = None,
    benchmark_config_path: str = None,
    provider_override: str = None,
    seed: int = 42,
    risk_free_rate: float = 0.04,
    start_date_override: str = None,
    end_date_override: str = None
) -> None:
    print("\n" + "=" * 80)
    print("  QUANTITATIVE PORTFOLIO WEALTH SIMULATOR & TAX-ALPHA RESEARCH ENGINE")
    print("  PHASES 1 - 12: Multi-Currency, Analytics, Backtest, MC, Optimization, Stress & Tax")
    print("=" * 80 + "\n")

    if config_path is None:
        config_path = os.path.join(CURRENT_DIR, "config", "portfolio.yaml")
    if tax_config_path is None:
        tax_config_path = os.path.join(CURRENT_DIR, "config", "tax.yaml")

    print(f" Loading Portfolio Configuration from: {config_path}")
    config = PortfolioConfig.from_yaml(config_path)
    portfolio = Portfolio(config)
    currency_sym = "$" if portfolio.base_currency.upper() == "USD" else portfolio.base_currency + " "

    # 3. Determine Provider early so ^IRX yield can be dynamically ingested
    target_provider = provider_override or config.market_data_settings.get("default_provider", "yfinance")
    if target_provider.lower() == "yfinance":
        try:
            import yfinance
            provider = YFinanceProvider(auto_adjust=False)
            provider_desc = "Live Yahoo Finance Provider initialized."
        except ImportError:
            provider = MockMarketDataProvider(seed=seed)
            provider_desc = "Mock Provider initialized (yfinance fallback)."
    else:
        provider = MockMarketDataProvider(seed=seed)
        provider_desc = f"Deterministic Mock Provider initialized with seed={seed}."

    # Ingest 3-Month Treasury Bill Yield (^IRX) dynamically to measure historical risk-free rate
    s_date_init = start_date_override or config.market_data_settings.get("start_date", "2016-01-01")
    irx_mean_bull, irx_mean_full = fetch_irx_yield_averages(provider, target_provider, start_date=s_date_init, end_date=end_date_override)

    print(f"    * Portfolio Name:   '{portfolio.name}'")
    print(f"    * Base Currency:    {portfolio.base_currency} ({currency_sym.strip()})")
    print(f"    * Risk-Free Rate:   {risk_free_rate*100:.2f}% per year (Constant baseline assumption)")
    print(f"      Note on Historical Sharpe: Constant 4.0% reflects current terminal cash yield proxy; actual 2016-2021 T-Bill yields (^IRX)")
    print(f"      averaged {irx_mean_bull:.2f}% (full period {irx_mean_full:.2f}%), meaning historical Sharpe ratios evaluated against constant 4.0% are conservatively understated.")

    # 2. Display Allocation Breakdown
    print("\n Configured Asset Allocation Breakdown:")
    alloc_df = portfolio.get_allocation_breakdown()
    print(alloc_df.to_string(index=False))

    eff_weights = config.get_effective_ticker_weights()
    ticker_currs = config.get_effective_ticker_currencies()
    print("\n    Effective Market Ticker Weights (Sum = 100%):")
    for t, w in eff_weights.items():
        curr = ticker_currs.get(t, "USD")
        print(f"    * {t:<10} : {w*100:>5.2f}% (Quote: {curr})")

    # 3. Display Provider
    print(f"\n[3] Ingesting Market Data using provider: '{target_provider}'")
    print(f"    * {provider_desc}")

    # 4. Ingest, Validate & Align Data with Multi-Currency Engine
    print("\n[4] Validating, Aligning & Multi-Currency Converting Time Series...")
    report = portfolio.load_data(
        provider,
        start_date=start_date_override,
        end_date=end_date_override
    )
    print("\n" + report.summary_text() + "\n")

    print("[4.1] Multi-Currency Base Conversion & FX Impact Audit:")
    curr_summary = portfolio.get_currency_exposure_summary()
    print(curr_summary.to_string())
    print()

    # 5. Comprehensive Risk Analytics (Phase 2 - Weekly Calendar-Aligned)
    print("[5] Phase 2: Historical Risk, Tail-Risk & Distribution Analytics (Weekly Calendar-Aligned, 52 wks/yr):")
    print("    [*] Calibrated on Friday-to-Friday weekly compounding (52 periods/year) to eliminate")
    print("        artificial weekend zero-return dilution caused by combining 24/7 crypto with 5-day equities/bonds.")
    risk_df = portfolio.get_comprehensive_risk_report(risk_free_rate=risk_free_rate, frequency="weekly")

    fmt_risk = risk_df.copy()
    pct_cols = [c for c in fmt_risk.columns if "%" in c]
    ratio_cols = [c for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Skewness", "Excess Kurtosis"] if c in fmt_risk.columns]

    for col in pct_cols:
        fmt_risk[col] = fmt_risk[col].map("{:+.2f}%".format)
    for col in ratio_cols:
        fmt_risk[col] = fmt_risk[col].map("{:.3f}".format)

    print(fmt_risk.to_string())

    # 6. Empirical & Regime-Switching Correlation Matrices (Weekly Calendar-Aligned)
    print("\n[6] Phase 2: Empirical Correlation Matrix (Weekly Calendar-Aligned):")
    corr_matrix = portfolio.get_correlation_matrix(frequency="weekly")
    print(format_correlation_heatmap_text(corr_matrix))

    print("\n[6.1] Dynamic Regime-Switching Correlation (Normal vs Crisis States - Weekly Calibrated):")
    weekly_assets = portfolio.get_weekly_asset_returns()
    regime_corr_res = calculate_regime_switching_correlation(
        weekly_assets,
        method="volatility_threshold",
        threshold=75.0,
        rolling_window=4,
        periods_per_year=52.0
    )
    print(regime_corr_res.summary_text())

    # 7. Historical Backtesting Engine (Phase 3)
    print("\n[7] Phase 3: Historical Portfolio Backtesting (USD Real vs Nominal Wealth)...")
    bt_settings = config.backtest_settings or config.market_data_settings.get("backtest", {})
    bt_config = BacktestConfig(
        initial_capital=float(bt_settings.get("initial_capital", 10_000.0)),
        monthly_contribution=float(bt_settings.get("monthly_contribution", 500.0)),
        contribution_growth_rate=float(bt_settings.get("contribution_growth_rate", 0.10)),
        rebalance_frequency=str(bt_settings.get("rebalance_frequency", "quarterly")),
        rebalance_threshold=float(bt_settings.get("rebalance_threshold", 0.05)),
        transaction_cost_pct=float(bt_settings.get("transaction_cost_pct", 0.001)),
        annual_inflation_rate=float(bt_settings.get("annual_inflation_rate", 0.03)),
        currency_symbol=currency_sym.strip(),
        inflow_rebalancing=True  # Default to tax-efficient cash flow steering
    )

    backtester = Backtester()
    bt_result = backtester.run(portfolio, bt_config)
    print("\n" + "=" * 55 + " BACKTEST PERFORMANCE SUMMARY " + "=" * 55)
    print(bt_result.summary_table().to_string())
    print("=" * 140)

    # 7.1 Rebalancing Strategy Comparison (Standard vs Tax-Efficient Inflow Steering)
    print("\n[7.1] Rebalancing Strategy Comparison (Calendar vs Inflow Steering vs Zero-Sell):")
    df_rebal_comp = backtester.compare_rebalancing_strategies(portfolio, bt_config)
    print(df_rebal_comp.to_string())

    # 7.2 Portfolio Construction & Risk Parity Comparison (Phase 7 - Weekly Calibrated)
    print("\n[7.2] Phase 7: Strategy Allocation Comparison (Fixed vs Equal vs MinVar vs Risk Parity - Weekly Calibrated)...")
    print("    [*] Covariance and risk contributions calibrated on weekly returns (52 wks/yr) to remove calendar distortion.")
    cov_matrix_weekly = portfolio.get_covariance_matrix(frequency="weekly").loc[list(config.assets.keys()), list(config.assets.keys())]
    optimizer = PortfolioOptimizer(cov_matrix_weekly)
    fixed_weights = config.get_top_level_allocations()
    comp_df = optimizer.compare_strategies(
        asset_returns=portfolio.get_weekly_asset_returns(),
        fixed_weights=fixed_weights,
        risk_free_rate=risk_free_rate,
        periods_per_year=52.0
    )
    print(comp_df.to_string())

    # Dynamically extract empirical figures from Phase 2 and Phase 7.2 to eliminate stale hardcoded text
    p_cagr = float(risk_df.loc["Portfolio", "CAGR (%)"]) if "Portfolio" in risk_df.index else 0.0
    p_vol = float(risk_df.loc["Portfolio", "Ann. Volatility (%)"]) if "Portfolio" in risk_df.index else 0.0
    fw_row = comp_df.loc["Fixed-Weight"] if "Fixed-Weight" in comp_df.index else None
    fw_cagr = float(fw_row["Ann. Return (%)"]) if fw_row is not None else 0.0
    fw_vol = float(fw_row["Ann. Volatility (%)"]) if fw_row is not None else 0.0
    fw_mdd = float(fw_row["Max Drawdown (%)"]) if fw_row is not None else 0.0

    print("\n    [!] Rebalancing Methodology & Assumption Note:")
    print("        * Phase 7.2 assumes STATIC WEEKLY REBALANCING (weights reset back to target at every weekly Friday close).")
    print("        * Phase 2 & 10 evaluate the DAILY-REBALANCED portfolio resampled weekly (Friday close compounding).")
    print(f"        * Due to Bitcoin's high intra-week daily volatility, daily rebalancing locks in gains daily (CAGR {p_cagr:+.2f}%, Vol {p_vol:.2f}%),")
    print(f"          whereas weekly rebalancing lets intra-week drift occur before Friday rebalancing (CAGR {fw_cagr:+.2f}%, Vol {fw_vol:.2f}%, MDD {fw_mdd:.2f}%).")
    print("        * Phase 3 backtest reflects QUARTERLY rebalancing with a 5% deviation tolerance band.")

    # 8. Monte Carlo Simulation Engine (Phase 4 & 5)
    print("\n[8] Phase 4 & 5: Monte Carlo Probabilistic Simulation Engine (10,000 Paths)...")
    print("    [!] WARNING: Monte Carlo simulation is not yet recalibrated (still using raw daily trading data).")
    print("        The simulation results below are internal diagnostics and NOT INCLUDED in the primary investment report.")
    sim_engine = MonteCarloEngine()

    mc_cfg_parametric = SimulationConfig(
        n_simulations=10_000,
        horizon_years=20,
        time_steps_per_year=12,
        model_type=SimulationModel.PARAMETRIC_STUDENT_T.value,
        degrees_of_freedom=5,
        random_seed=seed,
        initial_capital=bt_config.initial_capital,
        monthly_contribution=bt_config.monthly_contribution,
        contribution_growth_rate=bt_config.contribution_growth_rate,
        annual_inflation_rate=bt_config.annual_inflation_rate,
        rebalance_frequency="quarterly",
        currency_symbol=currency_sym.strip()
    )
    sim_res_parametric = sim_engine.run_simulation(portfolio, mc_cfg_parametric)

    print("\n--- Real Wealth Projections (Today's Purchasing Power, 3% Inflation Adj.) ---")
    df_proj_real = sim_res_parametric.get_horizon_projection_table(horizons=[5, 10, 15, 20], real=True)
    print(df_proj_real.to_string())

    dd_stats = sim_res_parametric.get_drawdown_percentiles(real=True)
    print("\n--- Simulated Maximum Drawdown Distribution Across 10,000 Paths ---")
    print(f"    * Severe Tail Drawdown (P5)      : {dd_stats['MDD_P5']*100:.2f}% (95% worst-case bear market)")
    print(f"    * Adverse Tail Drawdown (P10)    : {dd_stats['MDD_P10']*100:.2f}%")
    print(f"    * Median Path Max Drawdown (P50) : {dd_stats['MDD_P50']*100:.2f}%")
    print(f"    * Mild Drawdown (P90)            : {dd_stats['MDD_P90']*100:.2f}%")
    print(f"    * Minimum Drawdown (P95)         : {dd_stats['MDD_P95']*100:.2f}%")

    print("\n" + sim_res_parametric.get_p5_p95_conclusion(horizons=[5, 10, 20], real=True))

    # 9. Tax Engine & CSPX Tax-Alpha Analysis (Phase 8)
    print("\n[9] Phase 8: Tax Engine & S&P 500 CSPX Tax-Alpha Analysis...")
    tax_engine = TaxEngine(tax_config_path)
    print("\n" + tax_engine.get_audit_report().summary_text() + "\n")

    alpha_engine = TaxAlphaEngine(tax_engine)
    cspx_comparison = alpha_engine.evaluate_cspx_advantage(
        initial_investment=bt_config.initial_capital * config.assets["CSPX"].allocation,
        annual_gross_return=0.10,
        dividend_yield=0.0150,
        monthly_contribution=bt_config.monthly_contribution * config.assets["CSPX"].allocation,
        contribution_growth_rate=bt_config.contribution_growth_rate,
        currency_symbol=currency_sym.strip()
    )

    print(">>> S&P 500 STRUCTURAL COMPARISON: CSPX (Ireland UCITS) vs VOO/SPY (US Domiciled)")
    print("    * CSPX: Ireland UCITS Accumulating ETF (15% US-Ireland Treaty WHT at fund level, 0% Irish WHT, No US Estate Tax)")
    print("    * VOO : US Domiciled Distributing ETF (30% US NRA Statutory WHT, Taxable Cash Distribution Drag, US Estate Tax Exposure)\n")

    df_tax_alpha = cspx_comparison.compute_horizon_tax_alpha(horizons=[5, 10, 15, 20, 30])
    print(df_tax_alpha.to_string())

    print("\n--- Tax Alpha Sensitivity Grid (15-Year Horizon, Net Dollar Advantage) ---")
    df_sens = cspx_comparison.compute_sensitivity_matrix(horizon_years=15)
    print(df_sens.to_string())

    # 10. Macro Factor Stress Testing (Phase 9)
    print("\n[10] Phase 9: Macro Factor Stress Testing (Severe Factor Shocks)...")
    for sc_name in ["EQUITY_CRASH_2008", "CRYPTO_WINTER", "EM_CURRENCY_CRISIS_USD_SPIKE", "STAGFLATION_INFLATION_SHOCK"]:
        print(f"\n--- Scenario: {sc_name} ---")
        df_sc = StressTestEngine.apply_scenario(
            weights=fixed_weights,
            initial_wealth=100_000.0,
            scenario_name=sc_name,
            currency_symbol=currency_sym.strip()
        )
        print(df_sc.to_string())

    # 11. Benchmark-Relative Performance Analysis (Phase 10)
    benchmark_cfg_path = benchmark_config_path or os.path.join(CURRENT_DIR, "config", "benchmark.yaml")
    if os.path.exists(benchmark_cfg_path):
        print("\n[11] Phase 10: Benchmark-Relative Performance Analysis (60/40 Global Eq/Bond)...")
        bench_config = PortfolioConfig.from_yaml(benchmark_cfg_path)
        bench_portfolio = Portfolio(bench_config)
        bench_portfolio.load_data(
            provider,
            start_date=start_date_override,
            end_date=end_date_override
        )
        bench_weights = bench_config.get_top_level_allocations()
        from src.analytics.returns import calculate_portfolio_returns as calc_port_rets
        bench_returns = calc_port_rets(bench_portfolio.asset_returns, bench_weights, rebalanced=True)

        # Ingest 10-Year Treasury Yield (^TNX) with strict validation & fail-fast error handling
        import datetime
        import time
        today_date = datetime.date.today()
        # Find last fully closed global trading day (yesterday or previous Friday)
        last_complete_date = today_date - datetime.timedelta(days=1)
        while last_complete_date.weekday() >= 5:
            last_complete_date -= datetime.timedelta(days=1)
        last_complete_str = last_complete_date.strftime("%Y-%m-%d")

        s_date = start_date_override or config.market_data_settings.get("start_date", "2016-01-01")
        raw_end = end_date_override or config.market_data_settings.get("end_date", last_complete_str)
        if not raw_end or str(raw_end).strip().lower() in ("today", "now", "none", ""):
            resolved_end = last_complete_str
        else:
            try:
                parsed_e = datetime.date.fromisoformat(str(raw_end).strip())
                if parsed_e >= today_date:
                    resolved_end = last_complete_str
                else:
                    resolved_end = str(raw_end).strip()
            except Exception:
                resolved_end = last_complete_str

        is_historical_end = (resolved_end != today_date.strftime("%Y-%m-%d"))
        tnx_end_arg = resolved_end if is_historical_end else None

        print(f"    * Ingesting 10-Year Treasury Yield (^TNX) from {s_date} to {resolved_end}...")
        if target_provider.lower() == "yfinance":
            try:
                import yfinance as yf
                tnx_series = None

                # Method 1: yf.Ticker('^TNX').history (most resilient for CBOE index tickers)
                for attempt in range(1, 4):
                    try:
                        tk = yf.Ticker("^TNX")
                        hist = tk.history(start=s_date, end=tnx_end_arg, auto_adjust=False)
                        if not hist.empty:
                            col = "Adj Close" if "Adj Close" in hist.columns else "Close"
                            if col in hist.columns:
                                s = hist[col].dropna()
                                if len(s) >= 10:
                                    tnx_series = s
                                    break
                    except Exception:
                        pass
                    time.sleep(0.5)

                # Method 2: yf.download with start only (omits end to prevent future-date CBOE error)
                if tnx_series is None or len(tnx_series) < 10:
                    try:
                        tnx_df = yf.download("^TNX", start=s_date, end=tnx_end_arg, progress=False)
                        if not tnx_df.empty:
                            if isinstance(tnx_df.columns, pd.MultiIndex):
                                level_0 = tnx_df.columns.get_level_values(0)
                                if "Adj Close" in level_0:
                                    s = tnx_df["Adj Close"]
                                elif "Close" in level_0:
                                    s = tnx_df["Close"]
                                else:
                                    s = tnx_df.iloc[:, 0]
                                if isinstance(s, pd.DataFrame):
                                    s = s.iloc[:, 0]
                            else:
                                col = "Adj Close" if "Adj Close" in tnx_df.columns else "Close"
                                s = tnx_df[col]
                            s = s.dropna()
                            if len(s) >= 10:
                                tnx_series = s
                    except Exception:
                        pass

                # Method 3: Fallback using period='max' filtered by start_date
                if tnx_series is None or len(tnx_series) < 10:
                    try:
                        tk = yf.Ticker("^TNX")
                        hist_max = tk.history(period="max", auto_adjust=False)
                        if not hist_max.empty:
                            col = "Adj Close" if "Adj Close" in hist_max.columns else "Close"
                            s = hist_max[col].dropna()
                            if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                                s.index = s.index.tz_localize(None)
                            s = s.loc[s_date:]
                            if len(s) >= 10:
                                tnx_series = s
                    except Exception:
                        pass

                # Method 4: Offline fallback to persistent local cache
                if tnx_series is None or len(tnx_series) < 10:
                    cached_tnx = YFinanceProvider._load_from_cache("^TNX", s_date, resolved_end)
                    if cached_tnx is not None and len(cached_tnx) >= 10:
                        tnx_series = cached_tnx
                        logger.warning(f"Using local cached data for 10Y Treasury yield (^TNX) ({len(tnx_series)} bars).")

                if tnx_series is None or len(tnx_series) < 10:
                    raise RuntimeError(
                        f"Yahoo Finance returned empty or insufficient data for '^TNX' "
                        f"between {s_date} and {resolved_end}."
                    )

                # Save successful series to cache
                YFinanceProvider._save_to_cache("^TNX", tnx_series)

                if isinstance(tnx_series.index, pd.DatetimeIndex) and tnx_series.index.tz is not None:
                    tnx_series.index = tnx_series.index.tz_localize(None)
                tnx_series = tnx_series.dropna()
                tnx_weekly = tnx_series.resample("W-FRI").last().dropna()
                yield_changes = tnx_weekly.diff().dropna()
                yield_changes.name = "dYield_10Y_weekly"
                print(f"    * 10Y Treasury weekly yield (^TNX) loaded successfully ({len(yield_changes)} weekly changes).")
            except Exception as err:
                raise RuntimeError(
                    f"CRITICAL ERROR: Failed to download 10-Year Treasury yield (^TNX) from Yahoo Finance: {err}. "
                    f"Yield sensitivity calculation aborted."
                ) from err
        else:
            tnx_mock = provider.fetch_history(["^TNX"], start_date=s_date, end_date=resolved_end)
            if not tnx_mock.empty and "^TNX" in tnx_mock.columns:
                tnx_series = tnx_mock["^TNX"].dropna()
                tnx_weekly = tnx_series.resample("W-FRI").last().dropna()
                yield_changes = tnx_weekly.diff().dropna()
                yield_changes.name = "dYield_10Y_weekly"
                print(f"    * 10Y Treasury synthetic weekly yield (^TNX) loaded ({len(yield_changes)} weekly changes).")
            else:
                raise RuntimeError("Mock provider failed to generate ^TNX yield data.")

        # 3-Month Treasury Bill Yield (^IRX) baseline loaded dynamically at initialization
        print(f"    * 3M Treasury Bill yield (^IRX) historical baseline: 2016-2021 mean = {irx_mean_bull:.2f}%, full mean = {irx_mean_full:.2f}%.")

        has_bonds = any(a.type.lower() == "bond" for a in portfolio.config.assets.values())
        bnd_rets = bench_portfolio.asset_returns.get("GLOBAL_BOND")
        if bnd_rets is None and "BND" in bench_portfolio.asset_returns.columns:
            bnd_rets = bench_portfolio.asset_returns["BND"]

        top_allocs = portfolio.config.get_top_level_allocations()

        bench_report = compute_benchmark_report(
            portfolio_returns=portfolio.portfolio_returns,
            benchmark_returns=bench_returns,
            yield_changes=yield_changes,
            risk_free_rate=risk_free_rate,
            benchmark_name=bench_config.portfolio_name,
            has_bond_allocation=has_bonds,
            bnd_returns=bnd_rets,
            asset_returns=portfolio.asset_returns,
            weights=top_allocs,
            irx_mean_bull=irx_mean_bull,
            irx_mean_full=irx_mean_full,
        )
        print("\n" + bench_report.summary_text())

        print("\n[11.1] Benchmark Comparison Summary Table:")
        print(bench_report.summary_table().to_string())
    else:
        print(f"\n[11] Benchmark config not found at '{benchmark_cfg_path}'. Skipping benchmark analysis.")

    # 12. Walk-Forward Out-of-Sample Validation (Phase 11)
    print("\n[12] Phase 11: Walk-Forward Out-Of-Sample Validation (Rolling Windows)...")
    print("    [!] WARNING: Walk-Forward validation is not yet recalibrated (still using raw daily trading windows).")
    print("        The validation results below are internal diagnostics and NOT INCLUDED in the primary investment report.")
    if len(portfolio.asset_returns) >= 756 + 252:
        wf_validator = WalkForwardValidator(portfolio.asset_returns, periods_per_year=portfolio.periods_per_year, risk_free_rate=risk_free_rate)
        df_wf = wf_validator.run_walk_forward(
            strategy="fixed",
            fixed_weights=fixed_weights,
            train_days=756,  # 3-year train
            test_days=252,   # 1-year test
            step_days=252
        )
        print(df_wf.to_string())
    else:
        print("    [!] Insufficient history for 4-year rolling walk-forward validation.")

    print("\n" + "=" * 80)
    print("  Quantitative Research Engine Execution Completed Successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Portfolio Simulator Pipeline (USD)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(CURRENT_DIR, "config", "portfolio.yaml"),
        help="Path to portfolio.yaml configuration file"
    )
    parser.add_argument(
        "--tax-config",
        type=str,
        default=os.path.join(CURRENT_DIR, "config", "tax.yaml"),
        help="Path to tax.yaml configuration file"
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default=os.path.join(CURRENT_DIR, "config", "benchmark.yaml"),
        help="Path to benchmark.yaml configuration file (default: 60/40 Global Equity/Bond)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["mock", "yfinance"],
        help="Override data provider (default: read from config/portfolio.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mock data provider (default: 42)"
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annual risk-free rate assumption (default: 0.04 or 4%)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override start date (format: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Override end date (format: YYYY-MM-DD or 'today')"
    )

    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        tax_config_path=args.tax_config,
        benchmark_config_path=args.benchmark_config,
        provider_override=args.provider,
        seed=args.seed,
        risk_free_rate=args.rf,
        start_date_override=args.start_date,
        end_date_override=args.end_date
    )
