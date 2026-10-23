"""
Unit tests for Tax Engine, Tax Drag, Legal Transparency, and CSPX Tax Alpha (Phase 8 & 9).
"""

import unittest
import os
import pandas as pd
from src.tax.tax_engine import TaxEngine, AssetTaxConfig, TaxCategory
from src.tax.tax_alpha import TaxAlphaEngine, CSPXTaxAlphaComparison


class TestTaxEngine(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "tax.yaml"
        )
        self.tax_engine = TaxEngine(self.config_path)

    def test_load_tax_config_and_audit_report(self):
        audit = self.tax_engine.get_audit_report()
        self.assertEqual(audit.country, "Indonesia")
        self.assertIn("CSPX", audit.asset_rules)
        self.assertIn("VOO_BASELINE", audit.asset_rules)
        self.assertIn("BTC", audit.asset_rules)
        self.assertIn("BANK", audit.asset_rules)

        cspx_cfg = audit.asset_rules["CSPX"]
        self.assertEqual(cspx_cfg.tax_category, "FACT")
        self.assertEqual(cspx_cfg.verification_status, "VERIFIED")
        self.assertEqual(cspx_cfg.dividend_withholding_tax_rate, 0.15)

        summary_txt = audit.summary_text()
        self.assertIn("TAX SOURCE & LEGAL TRANSPARENCY AUDIT", summary_txt)
        self.assertIn("US-Ireland Double Taxation Treaty", summary_txt)

    def test_effective_dividend_drag_calculation(self):
        cspx_cfg = self.tax_engine.asset_tax_configs["CSPX"]
        self.assertAlmostEqual(cspx_cfg.effective_annual_dividend_tax_drag, 0.0150 * 0.15, places=5)

        voo_cfg = self.tax_engine.asset_tax_configs["VOO_BASELINE"]
        self.assertAlmostEqual(voo_cfg.effective_annual_dividend_tax_drag, 0.0150 * 0.30, places=5)

    def test_after_tax_return_adjustment(self):
        gross_r = 0.10
        net_cspx_r = self.tax_engine.calculate_asset_after_tax_return(gross_r, "CSPX", periods_per_year=1.0)
        expected_net = 0.10 - (0.0150 * 0.15)
        self.assertAlmostEqual(net_cspx_r, expected_net, places=5)

    def test_cspx_tax_alpha_comparison_table(self):
        comparison = CSPXTaxAlphaComparison(
            initial_investment=10_000.0,
            annual_sp500_gross_return=0.10,
            dividend_yield=0.015,
            monthly_contribution=500.0,
            contribution_growth_rate=0.10,
            currency_symbol="$"
        )
        df_alpha = comparison.compute_horizon_tax_alpha(horizons=[5, 10, 20, 30])
        self.assertEqual(len(df_alpha), 4)
        self.assertIn("5 Years", df_alpha.index)
        self.assertIn("20 Years", df_alpha.index)
        self.assertIn("CSPX (Ireland UCITS)", df_alpha.columns)
        self.assertIn("VOO (US Domiciled)", df_alpha.columns)
        self.assertIn("Tax Alpha (Net Gain)", df_alpha.columns)
        self.assertIn("Annualized Alpha", df_alpha.columns)

        self.assertIn("+22.5 bps/yr", df_alpha.loc["5 Years", "Annualized Alpha"])

    def test_tax_alpha_engine_factory(self):
        alpha_engine = TaxAlphaEngine(self.tax_engine)
        comp = alpha_engine.evaluate_cspx_advantage(initial_investment=50_000.0)
        self.assertIsInstance(comp, CSPXTaxAlphaComparison)
        self.assertEqual(comp.initial_investment, 50_000.0)


if __name__ == "__main__":
    unittest.main()