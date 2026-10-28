# Portfolio Backtest Engine

Multi-asset portfolio simulator with DCA, rebalancing, tax analysis, and benchmark comparison. Built from scratch with NumPy/Pandas/SciPy — no external quant libraries.

---

## Executive Overview (for HR / Hiring Managers)

Simulates investing $10K initial capital + $500/month DCA across a multi-asset portfolio (default: 45% S&P 500 / 25% BTC / 25% BBCA / 5% Gold) over 8+ years of historical data.

| Key Question | Methodology | Core Finding |
|:---|:---|:---|
| **Return & Risk** | Calendar Compound Annual Growth Rate (CAGR) & Weekly Volatility | Portfolio CAGR ~28.6% with 21.7% annualized weekly volatility |
| **Benchmark Comparison** | Weekly Active Return vs 60/40 Benchmark (60% VT / 40% BND) | Active return +20.05%p with 18.0% weekly tracking error (BTC ±66%, rebalancing ±22%) |
| **Yield Sensitivity** | Weekly OLS regression on 10Y Treasury yield (^TNX) changes | BND sanity check ($\beta \approx -0.055$, implied duration ~5.5 yrs); Portfolio has zero direct duration exposure |
| **Tax Efficiency** | UCITS Ireland (CSPX) vs US Domicile (VOO) Withholding Tax | CSPX generates +22.5 bps/year tax alpha via 15% US-Ireland treaty rate |
| **Crash Protection** | Dynamic regime-switching correlation (4-week rolling vol) | Pairwise correlation higher in volatile periods (+0.08 to +0.34); Forbes & Rigobon (2002) selection bias noted |
| **Turnover Reduction** | Tax-efficient cash flow steering (inflow-only rebalancing) | Reduces portfolio sales by ~60% while maintaining target weights |
| **Active Attribution** | Simplified Active Contribution Breakdown ($w_i \times (\text{CAGR}_i - \text{CAGR}_b)$) | BTC allocation drives ±66%, daily rebalancing bonus ±22% |

> [!NOTE]
> **Diagnostic Module Status:** Monte Carlo simulation (Phase 4-5) and Walk-Forward validation (Phase 9) currently operate on raw daily trading data (pending recalibration to the 52-week calendar). Therefore, both modules serve as internal diagnostics and are intentionally excluded from this primary investment summary.

---

## Core Analytics & Methodologies

### 1. Standardized Return Methodology: Calendar CAGR
To prevent calendar-induced distortions when combining 24/7 assets (Bitcoin trades 365 days/year) with traditional equities/bonds (252 trading days/year), all annualized returns across the engine are strictly standardized on **Calendar Compound Annual Growth Rate (CAGR)**:
$$\text{CAGR} = \left(\frac{P_{\text{end}}}{P_{\text{start}}}\right)^{\frac{365.25}{\text{calendar\_days}}} - 1$$
- **Buy-and-Hold Portfolio CAGR:** ~28.5% (static mix compounding).
- **Strategy TWR (Time-Weighted Return):** ~31.5% (pure investment return capturing rebalancing/DCA bonus, net of fees).
- **Strategy MWR (XIRR):** Money-weighted internal rate of return reflecting contribution cash-flow timing.

### 2. Calendar Synchronization: Weekly Volatility & Tracking Error
- **Weekly Volatility:** Friday close-to-Friday close compounding ($\prod (1 + R_d) - 1$) properly aligns 7-day crypto markets with 5-day stock markets, preserving all calendar crypto returns without dropping weekends. All volatility, tracking error, and Information Ratio metrics are evaluated strictly on weekly frequency (52 wks/year) to prevent calendar distortion.
- **Weekly Tracking Error:** $\text{TE} = \sigma(R_{p,w} - R_{b,w}) \times \sqrt{52}$. Computed strictly from the active return difference.
- **Tracking Error Audit:** The engine reports the variance decomposition $\text{Var}(R_p - R_b) = \sigma_p^2 + \sigma_b^2 - 2\text{Cov}(p, b)$. Because benchmark variance approximately offsets twice the covariance with the crypto portfolio, TE naturally approaches portfolio volatility.
- **Active Return Attribution (NOT Alpha):** Outperformance over the benchmark is labeled **Active Return** ($R_p - R_b = +20.05\%p$), supported by an empirical asset attribution table. Total active return cleanly decomposes into:
  - **Asset Active Contributions:** $+15.64\%p$ total (Bitcoin $+13.23\%p$, Equities $+2.98\%p$, Real Estate $+0.19\%p$, Gold $-0.76\%p$).
  - **Rebalancing & Compounding Effect:** $+4.41\%p$ ($22.0\%$ share), as Portfolio CAGR ($28.66\%$) exceeds the weighted sum of standalone asset CAGRs ($24.25\%$) from continuous daily rebalancing (volatility harvesting / Shannon's demon).
  - **Key Driver Summary:** Total active return is driven by **BTC active allocation ($\pm 66\%$)** and the **rebalancing/compounding interaction ($\pm 22\%$)**, not stock-picking alpha. Subjective rating labels (e.g. "IR = GOOD") are intentionally omitted.
  - **Active Contribution Definition:** 'Active Contrib' is defined as Weight * (Asset CAGR - Benchmark CAGR). This is a simplified contribution analysis rather than a full Brinson attribution, because the 60/40 benchmark lacks direct counterparts for Bitcoin, Gold, or Indonesian Equities.
- **Rebalancing Frequency Clarification:**
  - *Phase 2 & 10:* Daily-rebalanced synthetic portfolio compounded to Friday weekly close (CAGR ~28.6%, Vol ~21.7%, MDD -32.0%).
  - *Phase 7.2:* Static weekly rebalancing executed at each Friday close (CAGR ~28.0%, Vol ~21.9%, MDD -32.5%), reflecting intra-week weight drift.
  - *Phase 3:* Quarterly event-driven cash-flow rebalancing with 5% deviation bands (TWR ~31.5%).

### 3. Weekly 10-Year Treasury Yield Sensitivity (^TNX) & BND Sanity Check
- **Harmonized Weekly Frequency:** Evaluates interest rate sensitivity via weekly OLS regression: $R_{w} = \alpha + \beta \cdot \Delta Y_{10Y, w} + \epsilon_w$, matching the weekly frequency used throughout risk analytics.
- **BND Sanity Check:** Regressing BND (US Aggregate Bonds) alone against weekly $\Delta Y_{10Y}$ yields $\beta = -0.0547$ ($R^2 = 79.4\%$, $p < 10^{-100}$), confirming an implied modified duration of **$5.47$ years** (consistent with theoretical duration of ~6.0 years).
- **Direct Exposure Interpretation:** The active portfolio displays near-zero yield sensitivity ($\beta = -0.0011, p = 0.92$), confirming that the portfolio **has no direct duration exposure (0% direct bonds)**.
- **Strict Error Handling:** Fails fast with an explicit error message if yield data fails to download (never reports false zero beta or p=1.0).

---

## Quick Start

```bash
pip install -r requirements.txt

# Live data from Yahoo Finance
python main.py

# Deterministic mock data (reproducible offline)
python main.py --provider mock

# Custom benchmark configuration
python main.py --benchmark-config config/benchmark.yaml

# Run test suite (113 unit tests)
python -m pytest
```

---

## Configuration Architecture

All parameters are externalized in `config/` YAML files:

| File | Purpose | Key Parameters |
|:---|:---|:---|
| `portfolio.yaml` | Target assets & allocation weights | 45% CSPX.L, 25% BTC-USD, 25% BBCA.JK, 5% GLD |
| `benchmark.yaml` | Reference benchmark portfolio | 60% VT (Global Equity), 40% BND (Bonds) |
| `tax.yaml` | Legal tax rates & treaty database | 15% US-Ireland WHT, 10% ID PPh Final, 0% CGT |

---

## Repository Structure

```
├── config/
│   ├── portfolio.yaml      # Portfolio allocation + backtest settings
│   ├── benchmark.yaml      # 60/40 reference benchmark
│   └── tax.yaml            # Tax rates + legal citations
├── src/
│   ├── analytics/           # returns, risk, correlation, benchmark, validation
│   ├── backtest/            # day-by-day event-driven backtester (TWR / MWR)
│   ├── data/                # yfinance provider, FX engine, data validator
│   ├── portfolio/           # allocation config, optimizer, orchestrator
│   ├── simulation/          # Monte Carlo (Gaussian / Student-t / Bootstrap)
│   └── tax/                 # tax engine, CSPX vs VOO tax alpha
├── tests/                   # 113 unit tests (100% passing)
├── main.py                  # Full quantitative execution pipeline
└── requirements.txt
```

---

## Assumptions & Disclosures

| Parameter | Value | Justification |
|:---|:---|:---|
| Initial Capital | $10,000 | Baseline initial wealth |
| Monthly DCA | $500 | Escalated +10%/year (salary growth assumption) |
| Rebalancing | Quarterly | 5% maximum deviation tolerance band |
| Transaction Cost | 10 bps (0.10%) | Brokerage fee and market slippage per trade |
| Annual Inflation | 3.0% | Real purchasing power deflation factor |
| Risk-Free Rate | 4.0% | Constant baseline assumption matching terminal cash yields. (Actual 2016-2021 T-Bill yields [^IRX] averaged 0.93% [full period 2.27%], dynamically loaded from market data, so historical Sharpe ratios evaluated at constant 4.0% are conservatively understated) |
| Monte Carlo | 10,000 paths | Multivariate Student-t (ν=5) with Cholesky drift correction (diagnostic prototype, excluded from final report) |
| Benchmark | 60% VT / 40% BND | Classic institutional global balanced benchmark |

---

## Limitations

1. **Survivorship & Hindsight Bias:** Asset selection reflects historical winners (Bitcoin, BCA).
2. **Frictionless Daily Rebalancing Bonus:** The $+4.41\%p$ rebalancing bonus assumes frictionless daily rebalancing without transaction costs, spreads, or taxes. In real implementations with quarterly rebalancing and transaction costs (Phase 3), this bonus is partially attenuated (a key discussion topic for quant interviews).
3. **Execution Realism:** Assumes execution at closing prices without intra-day order book depth.
4. **FX Treatment:** Multiplicative compounding with forward-fill on non-trading days; no bid-ask spread.
5. **Research Tool:** Designed for quantitative portfolio research and wealth strategy simulation; not an automated trading execution system.
6. **Uncalibrated Modules (Excluded from Report):** Monte Carlo simulation (Phase 4 & 5) and Walk-Forward validation (Phase 9) currently operate on raw daily trading data (without 7-day calendar recalibration). They are intentionally excluded from the primary investment report pending weekly recalibration.
