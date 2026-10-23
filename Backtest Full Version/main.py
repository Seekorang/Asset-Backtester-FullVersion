"""
Personal Portfolio Wealth Simulator & Tax-Alpha Research Engine
Integrated Quantitative Execution Script (Multi-Currency & USD Base Denominated).
"""

import sys
import os
import argparse
import logging
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


def run_pipeline(
    config_path: str = None,
    tax_config_path: str = None,
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
    print(f"    * Portfolio Name:   '{portfolio.name}'")
    print(f"    * Base Currency:    {portfolio.base_currency} ({currency_sym.strip()})")
    print(f"    * Risk-Free Rate:   {risk_free_rate*100:.2f}% per year")

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

    # 3. Determine Provider
    target_provider = provider_override or config.market_data_settings.get("default_provider", "yfinance")
    print(f"\n[3] Ingesting Market Data using provider: '{target_provider}'")
    
    if target_provider.lower() == "yfinance":
        try:
            import yfinance
            provider = YFinanceProvider(auto_adjust=False)
            print("    * Live Yahoo Finance Provider initialized.")
        except ImportError:
            print("    [!] yfinance not installed. Falling back to Mock Provider.")
            provider = MockMarketDataProvider(seed=seed)
    else:
        provider = MockMarketDataProvider(seed=seed)
        print(f"    * Deterministic Mock Provider initialized with seed={seed}.")

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

    # 5. Comprehensive Risk Analytics (Phase 2)
    print("[5] Phase 2: Historical Risk, Tail-Risk & Distribution Analytics:")
    risk_df = portfolio.get_comprehensive_risk_report(risk_free_rate=risk_free_rate)

    fmt_risk = risk_df.copy()
    for col in ["Ann. Return (%)", "CAGR (%)", "Ann. Volatility (%)", "Downside Vol (%)", "Max Drawdown (%)",
                "Daily VaR 95% (%)", "Daily CVaR 95% (%)", "Daily VaR 99% (%)", "Daily CVaR 99% (%)",
                "Best Year (%)", "Worst Year (%)"]:
        fmt_risk[col] = fmt_risk[col].map("{:+.2f}%".format)

    for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Skewness", "Excess Kurtosis"]:
        fmt_risk[col] = fmt_risk[col].map("{:.3f}".format)

    print(fmt_risk.to_string())

    # 6. Empirical & Regime-Switching Correlation Matrices
    print("\n[6] Phase 2: Empirical Correlation Matrix:")
    corr_matrix = portfolio.get_correlation_matrix()
    print(format_correlation_heatmap_text(corr_matrix))

    print("\n[6.1] Dynamic Regime-Switching Correlation (Normal vs Crisis States):")
    regime_corr_res = calculate_regime_switching_correlation(
        portfolio.asset_returns,
        method="volatility_threshold",
        threshold=75.0,
        rolling_window=21,
        periods_per_year=portfolio.periods_per_year
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

    # 7.2 Portfolio Construction & Risk Parity Comparison (Phase 7)
    print("\n[7.2] Phase 7: Strategy Allocation Comparison (Fixed vs Equal vs MinVar vs Risk Parity)...")
    cov_matrix = portfolio.get_covariance_matrix().loc[list(config.assets.keys()), list(config.assets.keys())]
    optimizer = PortfolioOptimizer(cov_matrix)
    fixed_weights = config.get_top_level_allocations()
    comp_df = optimizer.compare_strategies(
        asset_returns=portfolio.asset_returns,
        fixed_weights=fixed_weights,
        risk_free_rate=risk_free_rate,
        periods_per_year=portfolio.periods_per_year
    )
    print(comp_df.to_string())

    # 8. Monte Carlo Simulation Engine (Phase 4 & 5)
    print("\n[8] Phase 4 & 5: Monte Carlo Probabilistic Simulation Engine (10,000 Paths)...")
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

    # 11. Walk-Forward Out-of-Sample Validation (Phase 9)
    print("\n[11] Phase 9: Walk-Forward Out-Of-Sample Validation (Rolling Windows)...")
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
        provider_override=args.provider,
        seed=args.seed,
        risk_free_rate=args.rf,
        start_date_override=args.start_date,
        end_date_override=args.end_date
    )
