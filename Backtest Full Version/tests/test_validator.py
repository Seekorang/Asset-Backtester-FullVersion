"""
Unit tests for Data Quality Engine, Alignment, and Validation.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.validator import DataValidator, DataQualityReport


class TestDataValidator(unittest.TestCase):

    def setUp(self):
        self.validator = DataValidator(outlier_std_threshold=3.0)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        self.clean_df = pd.DataFrame({
            "ASSET_A": np.linspace(100, 150, 100),
            "ASSET_B": np.linspace(50, 70, 100),
        }, index=dates)

    def test_clean_data_validation(self):
        cleaned_df, report = self.validator.validate_and_align(self.clean_df)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.duplicate_count, 0)
        self.assertEqual(report.missing_counts["ASSET_A"], 0)
        self.assertEqual(len(report.warnings), 0)
        self.assertIn("DATA QUALITY AUDIT REPORT", report.summary_text())

    def test_duplicate_dates_handling(self):
        dup_row = pd.DataFrame({"ASSET_A": [199.0], "ASSET_B": [99.0]}, index=[self.clean_df.index[10]])
        df_dup = pd.concat([self.clean_df, dup_row])

        cleaned_df, report = self.validator.validate_and_align(df_dup)
        self.assertEqual(report.duplicate_count, 1)
        self.assertEqual(len(cleaned_df), len(self.clean_df))
        self.assertEqual(cleaned_df.loc[self.clean_df.index[10], "ASSET_A"], 199.0)

    def test_missing_data_forward_fill(self):
        df_missing = self.clean_df.copy()
        df_missing.iloc[20:25, 0] = np.nan

        cleaned_df, report = self.validator.validate_and_align(df_missing, alignment_strategy="forward_fill")
        self.assertFalse(cleaned_df.isna().any().any())
        self.assertEqual(report.missing_counts["ASSET_A"], 5)
        self.assertEqual(cleaned_df.iloc[20, 0], cleaned_df.iloc[19, 0])

    def test_missing_data_intersection(self):
        df_missing = self.clean_df.copy()
        df_missing.iloc[20:25, 0] = np.nan

        cleaned_df, report = self.validator.validate_and_align(df_missing, alignment_strategy="intersection")
        self.assertEqual(len(cleaned_df), 100 - 5)

    def test_outlier_detection(self):
        df_outlier = self.clean_df.copy()
        df_outlier.iloc[50, 0] = df_outlier.iloc[49, 0] * 2.0

        cleaned_df, report = self.validator.validate_and_align(df_outlier)
        self.assertGreater(len(report.outliers_detected["ASSET_A"]), 0)
        self.assertTrue(any("outlier" in w for w in report.warnings))

    def test_non_positive_price_detection(self):
        df_neg = self.clean_df.copy()
        # Mengubah harga menjadi negatif secara eksplisit pada ASSET_B
        df_neg.loc[df_neg.index[30], "ASSET_B"] = -5.0

        cleaned_df, report = self.validator.validate_and_align(df_neg)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.non_positive_counts["ASSET_B"], 1)
        self.assertTrue(any("non-positive" in w for w in report.warnings))

    def test_timezone_conversion(self):
        df_tz = self.clean_df.copy()
        df_tz.index = df_tz.index.tz_localize("UTC")
        cleaned_df, report = self.validator.validate_and_align(df_tz)
        self.assertIsNone(cleaned_df.index.tz)


if __name__ == "__main__":
    unittest.main()