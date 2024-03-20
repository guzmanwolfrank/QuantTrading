import unittest
import numpy as np
import yfinance as yf
import pandas as pd

def fetch_intraday_data(ticker_symbol):
    # Fetch intraday data using yfinance
    data = yf.download(ticker_symbol, period="60d", interval="15m").round(2)
    return data

def calculate_open_only_range_percentage(intraday_data, daily_data):
    # Calculate open-only range percentage
    open_only_range_percentage = np.mean(intraday_data['OpenOnlyRange'] / daily_data['DayRange'])
    return open_only_range_percentage

class TestIntradayDataAnalysis(unittest.TestCase):
    
    def setUp(self):
        # Set up any data or resources needed for the tests
        self.ticker_symbol = "MNQ=F"
    
    def test_fetch_intraday_data(self):
        # Test fetching intraday data
        data = fetch_intraday_data(self.ticker_symbol)
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
    
    def test_calculate_open_only_range_percentage(self):
        # Test calculation of open-only range percentage
        intraday_data = fetch_intraday_data(self.ticker_symbol)
        daily_data = yf.download(self.ticker_symbol, period="60d", interval="1d").round(2)
        
        # Calculate open-only range percentage
        open_only_range_percentage = calculate_open_only_range_percentage(intraday_data, daily_data)
        
        # Assert the calculated percentage is within a valid range (0-1)
        self.assertTrue(0 <= open_only_range_percentage <= 1)
        
        # Assert the calculation is accurate within a reasonable tolerance
        expected_percentage = np.mean(intraday_data['OpenOnlyRange'] / daily_data['DayRange'])
        self.assertAlmostEqual(open_only_range_percentage, expected_percentage, delta=0.01)

if __name__ == '__main__':
    unittest.main()
