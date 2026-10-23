"""
Unit tests for Market Data Provider Abstraction and Resampling.
"""

import unittest
import pandas as pd
from src.data.provider import MockMarketDataProvider, MarketDataProvider


class TestDataProvider(unittest.TestCase):

    def setUp(self):
        self.provider = MockMarketDataProvider(seed=123)
        self.tickers = ["CSPX.L", "BTC-USD", "BBCA.JK", "GLD"]
        self.start_date = "2022-01-01"
        self.end_date = "2022-12-31"

    def test_daily_fetch(self):
        df = self.provider.fetch_history(self.tickers, self.start_date, self.end_date, frequency="daily")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(list(df.columns), self.tickers)
        self.assertGreater(len(df), 200)
        self.assertTrue((df > 0).all().all())

    def test_weekly_resample(self):
        df_weekly = self.provider.fetch_history(self.tickers, self.start_date, self.end_date, frequency="weekly")
        self.assertIsInstance(df_weekly, pd.DataFrame)
        self.assertGreater(len(df_weekly), 40)
        self.assertLess(len(df_weekly), 60)

    def test_monthly_resample(self):
        df_monthly = self.provider.fetch_history(self.tickers, self.start_date, self.end_date, frequency="monthly")
        self.assertIsInstance(df_monthly, pd.DataFrame)
        self.assertEqual(len(df_monthly), 12)

    def test_reproducibility_with_seed(self):
        p1 = MockMarketDataProvider(seed=999)
        p2 = MockMarketDataProvider(seed=999)
        df1 = p1.fetch_history(["BTC-USD"], "2023-01-01", "2023-06-30")
        df2 = p2.fetch_history(["BTC-USD"], "2023-01-01", "2023-06-30")
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_frequency(self):
        with self.assertRaises(ValueError):
            self.provider.fetch_history(self.tickers, self.start_date, self.end_date, frequency="hourly")


if __name__ == "__main__":
    unittest.main()