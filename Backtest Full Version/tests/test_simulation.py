"""
Unit tests for Monte Carlo Simulation & Probabilistic Wealth Projections (Phase 4 & 5).
"""

import unittest
import numpy as np
import pandas as pd

from src.portfolio.allocation import PortfolioConfig
from src.portfolio.portfolio import Portfolio
from src.data.provider import MockMarketDataProvider
from src.simulation.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationModel, SimulationResult
from src.simulation.bootstrap import BootstrapType


class TestMonteCarloSimulation(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
            "portfolio_name": "Test Simulation Portfolio",
            "base_currency": "USD",
            "assets": {
                "CSPX": {"type": "equity_etf", "allocation": 0.45, "ticker": "CSPX.L"},
                "BTC": {"type": "crypto", "allocation": 0.25, "ticker": "BTC-USD"},
                "BANK": {"type": "equity", "allocation": 0.25, "ticker": "BBCA.JK"},
                "GOLD": {"type": "commodity_or_gold_proxy", "allocation": 0.05, "ticker": "GLD"}
            },
            "market_data": {
                "frequency": "daily",
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "calendar_alignment": "forward_fill"
            }
        }
        self.portfolio_config = PortfolioConfig.from_dict(self.config_dict)
        self.portfolio = Portfolio(self.portfolio_config)
        self.provider = MockMarketDataProvider(seed=42)
        self.portfolio.load_data(self.provider)

        self.engine = MonteCarloEngine()

    def test_parametric_gaussian_simulation(self):
        cfg = SimulationConfig(
            n_simulations=1_000,
            horizon_years=5,
            time_steps_per_year=12,
            model_type=SimulationModel.PARAMETRIC_GAUSSIAN.value,
            random_seed=42,
            initial_capital=10_000.0,
            monthly_contribution=500.0
        )
        res = self.engine.run_simulation(self.portfolio, cfg)
        self.assertIsInstance(res, SimulationResult)
        self.assertEqual(res.nominal_paths.shape, (1000, 61))
        self.assertEqual(res.real_paths.shape, (1000, 61))

        stats = res.get_terminal_percentiles(real=False)
        self.assertGreaterEqual(stats["P95"], stats["P90"])
        self.assertGreater(stats["P90"], stats["P50"])
        self.assertGreater(stats["P50"], stats["P10"])
        self.assertGreaterEqual(stats["P10"], stats["P5"])

        dd = res.get_drawdown_percentiles(real=False)
        self.assertLessEqual(dd["MDD_P5"], dd["MDD_P10"])
        self.assertLessEqual(dd["MDD_P10"], dd["MDD_P50"])
        self.assertLessEqual(dd["MDD_P50"], dd["MDD_P90"])
        self.assertLessEqual(dd["MDD_P90"], dd["MDD_P95"])

    def test_reproducibility_with_seed(self):
        cfg1 = SimulationConfig(n_simulations=500, horizon_years=3, random_seed=999)
        cfg2 = SimulationConfig(n_simulations=500, horizon_years=3, random_seed=999)
        res1 = self.engine.run_simulation(self.portfolio, cfg1)
        res2 = self.engine.run_simulation(self.portfolio, cfg2)
        np.testing.assert_allclose(res1.nominal_paths, res2.nominal_paths)

    def test_parametric_student_t_simulation(self):
        cfg = SimulationConfig(
            n_simulations=1_000,
            horizon_years=5,
            model_type=SimulationModel.PARAMETRIC_STUDENT_T.value,
            degrees_of_freedom=4,
            random_seed=123
        )
        res = self.engine.run_simulation(self.portfolio, cfg)
        self.assertEqual(res.nominal_paths.shape, (1000, 61))
        self.assertTrue((res.nominal_paths > 0).all())

    def test_block_bootstrap_simulation(self):
        cfg = SimulationConfig(
            n_simulations=1_000,
            horizon_years=5,
            model_type=SimulationModel.HISTORICAL_BOOTSTRAP.value,
            bootstrap_type=BootstrapType.CIRCULAR_BLOCK.value,
            block_size=6,
            random_seed=42
        )
        res = self.engine.run_simulation(self.portfolio, cfg)
        self.assertEqual(res.nominal_paths.shape, (1000, 61))

    def test_iid_bootstrap_simulation(self):
        cfg = SimulationConfig(
            n_simulations=1_000,
            horizon_years=5,
            model_type=SimulationModel.HISTORICAL_BOOTSTRAP.value,
            bootstrap_type=BootstrapType.IID.value,
            random_seed=42
        )
        res = self.engine.run_simulation(self.portfolio, cfg)
        self.assertEqual(res.nominal_paths.shape, (1000, 61))

    def test_horizon_projection_table(self):
        cfg = SimulationConfig(n_simulations=1_000, horizon_years=10)
        res = self.engine.run_simulation(self.portfolio, cfg)
        df_proj = res.get_horizon_projection_table(horizons=[5, 10], real=False)
        self.assertIn("5 Years", df_proj.index)
        self.assertIn("10 Years", df_proj.index)
        self.assertIn("P5 (Severe Bear)", df_proj.columns)
        self.assertIn("P10 (Pessimistic)", df_proj.columns)
        self.assertIn("P50 (Median)", df_proj.columns)
        self.assertIn("P90 (Optimistic)", df_proj.columns)
        self.assertIn("P95 (Strong Bull)", df_proj.columns)

    def test_convergence_table(self):
        cfg = SimulationConfig(n_simulations=2_000, horizon_years=5)
        res = self.engine.run_simulation(self.portfolio, cfg)
        df_conv = res.get_convergence_check_table(check_points=[500, 1000, 2000])
        self.assertEqual(len(df_conv), 3)
        self.assertIn("Terminal P5", df_conv.columns)
        self.assertIn("Terminal P50 (Median)", df_conv.columns)
        self.assertIn("Terminal P95", df_conv.columns)

    def test_p5_p95_conclusion(self):
        cfg = SimulationConfig(n_simulations=1_000, horizon_years=10, initial_capital=10_000.0, monthly_contribution=500.0)
        res = self.engine.run_simulation(self.portfolio, cfg)
        conclusion = res.get_p5_p95_conclusion(horizons=[5, 10], real=True)
        self.assertIsInstance(conclusion, str)
        self.assertIn("EXECUTIVE CONCLUSION", conclusion)
        self.assertIn("P5 TO P95", conclusion)
        self.assertIn("90% confidence corridor", conclusion)
        self.assertIn("Severe Downside", conclusion)
        self.assertIn("Median Baseline", conclusion)
        self.assertIn("Strong Bull", conclusion)
        self.assertIn("TAIL RISK & DRAWDOWN CAPACITY AUDIT", conclusion)

    def test_portfolio_run_monte_carlo_method(self):
        res = self.portfolio.run_monte_carlo(SimulationConfig(n_simulations=500, horizon_years=2))
        self.assertIsInstance(res, SimulationResult)


if __name__ == "__main__":
    unittest.main()
