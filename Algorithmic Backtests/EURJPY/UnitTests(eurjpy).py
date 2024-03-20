import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import oandapyV20.endpoints.instruments as instruments
import oandapyV20
from datetime import datetime

class TestHistoricalData(unittest.TestCase):

    @patch('oandapyV20.endpoints.instruments.InstrumentsCandles')
    @patch('oandapyV20.API')
    def test_get_historical_data(self, mock_api, mock_candles):
        # Mock response data from OANDA API
        mock_response_data = {
            'candles': [
                {'time': '2023-01-01T00:00:00.000000Z', 'mid': {'o': '1.2000', 'h': '1.2100', 'l': '1.1950', 'c': '1.2050'}}
                # Add more data as needed for your test cases
            ]
        }
        # Set up mock API response
        mock_api.request.return_value = mock_response_data

        # Set up mock candles endpoint
        mock_candles.return_value = MagicMock()

        # Define test inputs
        instrument = "EUR_USD"
        granularity = "D"
        start_date = "2022-01-01"
        end_date = "2023-01-01"

        # Call the function
        historical_data = get_historical_data(instrument, granularity, start_date, end_date)

        # Check if the API was called with the correct parameters
        mock_api.request.assert_called_once_with(mock_candles.return_value)

        # Check if the returned data has the correct format
        self.assertIsInstance(historical_data, pd.DataFrame)
        self.assertEqual(historical_data.index[0], pd.Timestamp('2023-01-01T00:00:00.000000Z'))
        self.assertAlmostEqual(historical_data['open'][0], 1.2000)
        self.assertAlmostEqual(historical_data['high'][0], 1.2100)
        self.assertAlmostEqual(historical_data['low'][0], 1.1950)
        self.assertAlmostEqual(historical_data['close'][0], 1.2050)


# Function to fetch historical data
def get_historical_data(instrument, granularity, start, end):
    OANDA_ACCESS_TOKEN = "access key"
    access_token = OANDA_ACCESS_TOKEN
    API_KEY = access_token
    api = oandapyV20.API(access_token=API_KEY, environment="practice")

    params = {
        "granularity": granularity,
        "from": start,
        "to": end,
    }
    request = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = api.request(request)
    data = response['candles']
    ohlc_data = [{'time': candle['time'], 'open': float(candle['mid']['o']), 'high': float(candle['mid']['h']),
                  'low': float(candle['mid']['l']), 'close': float(candle['mid']['c'])} for candle in data]
    df = pd.DataFrame(ohlc_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


if __name__ == '__main__':
    unittest.main()
