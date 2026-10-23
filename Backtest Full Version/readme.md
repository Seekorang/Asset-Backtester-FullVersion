# Quantitative Portfolio Research & Tax-Alpha Engine

A production-grade, mathematically rigorous Python research engine for multi-asset portfolio construction, currency-converted historical backtesting, fat-tailed Monte Carlo simulations, portfolio optimization, stress testing, and tax-alpha analysis.

---

## 1. Asset Universe & Target Allocation

The portfolio implements a barbell allocation balancing global equity growth, cash-flow banking equity, asymmetric digital store-of-value, and monetary inflation hedging:

| Asset Code | Name / Underlying | Asset Class | Target Weight | Quote Currency | Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **CSPX** | `CSPX.L` (LSE) | Equity ETF | **45.0%** | USD | iShares Core S&P 500 UCITS ETF (Acc, Ireland-domiciled) |
| **BTC** | `BTC-USD` | Crypto | **25.0%** | USD | Bitcoin Spot (high-volatility asymmetric convex driver) |
| **BANK** | `BBCA.JK` (IDX) | Equity | **25.0%** | IDR | PT Bank Central Asia Tbk (premier Indonesian banking equity) |
| **GOLD** | `GLD` (NYSE) | Commodity | **5.0%** | USD | SPDR Gold Shares (non-correlated flight-to-safety proxy) |

---

## 2. Key Architecture & Quantitative Features

- **Multi-Currency Engine (`src/data/currency.py`)**: Exact multiplicative compounding $(1+R_{\text{base}}) = (1+R_{\text{local}})(1+R_{\text{FX}})$ avoiding arithmetic approximation errors. Denominates all metrics into the portfolio base currency (`USD` or `IDR`).
- **Data Integrity & Imputation Tracking (`src/data/validator.py`)**: Forward-fill alignment without lookahead bias (`bfill` strictly removed), tracking imputation counts and flagging statistical return outliers ($|z| > 4.0$).
- **Dynamic Regime-Switching Correlation (`src/analytics/correlation.py`)**: Partitions historical market regimes into Normal (calm) vs Crisis (stress) states based on volatility thresholds or market drawdowns, computing conditional correlation matrices ($\mathbf{R}_{\text{normal}}$ vs $\mathbf{R}_{\text{crisis}}$) and quantifying asymmetric diversification breakdown ($\Delta \mathbf{R} = \mathbf{R}_{\text{crisis}} - \mathbf{R}_{\text{normal}}$).
- **Tax-Efficient Inflow Rebalancing (`src/backtest/backtester.py`)**: Steers incoming monthly contributions exclusively toward underweight assets during parabolic bull runs (e.g. Bitcoin), eliminating forced capital gains realizations, reducing transaction costs by up to 60%, and allowing wide momentum tolerance bands without disruptive selling.
- **Disentangled Cash-Flow Backtesting (`src/backtest/backtester.py`)**: Distinguishes true investment compound return from capital contributions via **True Time-Weighted Return (TWR CAGR)** alongside **Money-Weighted Return (MWR / XIRR)**, accounting for scheduled rebalancing, slippage, and fees.
- **Fat-Tailed Monte Carlo & $P_5 \to P_{95}$ Corridor (`src/simulation/monte_carlo.py`)**:
  - Multivariate Student-$t$ distribution calibrated with variance scaling $\sqrt{(\nu - 2)/\nu}$ and Ito's drift adjustment to preserve target historical covariance.
  - Generates empirical Block Bootstrap simulations.
  - Computes complete $P_5$ to $P_{95}$ 90% confidence intervals, milestone wealth projections (5Y, 10Y, 20Y), maximum drawdown distributions (`MDD_P5` to `MDD_P95`), and synthesizes an executive decision report.
- **Portfolio Optimization (`src/portfolio/optimizer.py`)**: Compares Fixed Target Weights against Equal Weight (1/N), Minimum Variance, and Equal Risk Contribution (ERC / Risk Parity) via SLSQP constrained optimization.
- **Tax-Alpha Engine (`src/tax/tax_engine.py`)**: Computes the structural compounding advantage of Ireland-domiciled UCITS ETFs (`CSPX.L`, 15% treaty WHT, 0% local WHT, 0% US Estate Tax) vs. US-domiciled distributing ETFs (`VOO`, 30% NRA statutory WHT, taxable distribution friction).
- **Stress-Testing & Out-of-Sample Validation (`src/analytics/validation.py`)**:
  - Macro factor stress testing: `EQUITY_CRASH_2008`, `CRYPTO_WINTER`, `EM_CURRENCY_CRISIS_USD_SPIKE`, and `STAGFLATION_INFLATION_SHOCK`.
  - Rolling 4-year Walk-Forward Out-of-Sample (OOS) testing.

---

## 3. Directory Structure

```text
quant-wealth-tax-engine/
├── config/
│   ├── portfolio.yaml          # Canonical portfolio & backtest configuration
│   └── tax.yaml                # Domicile & withholding tax parameters
├── src/
│   ├── analytics/
│   │   ├── correlation.py      # Pearson, Spearman, rolling correlation & covariance
│   │   ├── returns.py          # Arithmetic, log, cumulative, annualized CAGR
│   │   ├── risk.py             # Volatility, Sharpe, Sortino, VaR/CVaR (95%/99%), MDD
│   │   └── validation.py       # Stress testing & rolling walk-forward validation
│   ├── backtest/
│   │   └── backtester.py       # Cash-flow accounting (TWR CAGR, MWR/XIRR, rebalancing)
│   ├── data/
│   │   ├── currency.py         # Multiplicative FX base conversion
│   │   ├── provider.py         # MockMarketDataProvider for reproducible offline tests
│   │   ├── validator.py        # Calendar alignment, imputation tracking, audit report
│   │   └── yfinance_provider.py# Live market data provider via yfinance
│   ├── portfolio/
│   │   ├── allocation.py       # Configuration parser & validation dataclasses
│   │   ├── optimizer.py        # Equal Weight, Min Variance, Equal Risk Contribution
│   │   └── portfolio.py        # Portfolio state orchestrator
│   ├── simulation/
│   │   ├── bootstrap.py        # Circular block & IID bootstrap return generator
│   │   └── monte_carlo.py      # Monte Carlo engine, P5-P95 percentiles & conclusion
│   └── tax/
│       └── tax_engine.py       # UCITS vs US-domiciled tax-alpha comparative engine
├── tests/                      # Comprehensive 76-test automated verification suite
│   ├── test_allocation.py
│   ├── test_backtest.py
│   ├── test_correlation.py
│   ├── test_data_provider.py
│   ├── test_deterministic_benchmarks.py
│   ├── test_optimizer.py
│   ├── test_portfolio.py
│   ├── test_returns.py
│   ├── test_risk.py
│   ├── test_simulation.py
│   ├── test_tax.py
│   ├── test_validation.py
│   └── test_validator.py
├── main.py                     # Main execution pipeline
├── README.md                   # System documentation
└── requirements.txt            # Python dependencies
```

---

## 4. Installation & Quick Start

### 4.1 Prerequisites
Python 3.10+ (tested on Python 3.11.9).

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4.2 Run the Pipeline
Execute the full quantitative pipeline with the deterministic mock provider:
```bash
python main.py --provider mock
```

Execute with live historical market data via yfinance:
```bash
python main.py --provider yfinance
```

### 4.3 Run Test Suite
Run all 76 unit tests and deterministic benchmarks:
```bash
python -m pytest
```
