# utils/data_processor.py

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import warnings

class DataProcessor:
    """
    Data processing utility for trading system.
    Handles data loading, cleaning, validation, and preprocessing.
    
    Parameters:
    -----------
    fill_method : str
        Method to fill missing values ('ffill', 'bfill', or 'interpolate')
    min_periods : int
        Minimum number of periods required for calculations
    adjust_prices : bool
        Whether to adjust prices for splits and dividends
    """
    
    def __init__(self,
                 fill_method: str = 'ffill',
                 min_periods: int = 30,
                 adjust_prices: bool = True):
        
        self.fill_method = fill_method
        self.min_periods = min_periods
        self.adjust_prices = adjust_prices
        
    def load_data(self,
                  symbol: str,
                  start_date: Union[str, datetime],
                  end_date: Union[str, datetime],
                  interval: str = '1d') -> pd.DataFrame:
        """
        Load financial data from yfinance
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'EURUSD=X')
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        interval : str
            Data interval ('1m', '5m', '1h', '1d', etc.)
            
        Returns:
        --------
        pd.DataFrame
            Processed OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.adjust_prices
            )
            
            if len(data) < self.min_periods:
                raise ValueError(f"Insufficient data points. Got {len(data)}, need {self.min_periods}")
                
            return self.process_data(data)
            
        except Exception as e:
            raise Exception(f"Error loading data for {symbol}: {str(e)}")
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data frame
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            Processed data
        """
        # Make copy to avoid modifying original
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Validate data quality
        df = self._validate_data(df)
        
        # Add basic technical features
        df = self._add_basic_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Check for missing values
        if df.isnull().any().any():
            # Fill missing OHLC values
            for col in ['Open', 'High', 'Low', 'Close']:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(method=self.fill_method)
            
            # Fill missing volume with 0
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and consistency"""
        # Ensure proper OHLC relationship
        df = df[
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
        ]
        
        # Remove zero or negative prices
        df = df[
            (df['Open'] > 0) &
            (df['High'] > 0) &
            (df['Low'] > 0) &
            (df['Close'] > 0)
        ]
        
        # Sort index
        df = df.sort_index()
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features"""
        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Trading ranges
        df['True_Range'] = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift(1)),
            'lc': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        
        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_StdDev'] = df['Volume'].rolling(window=20).std()
            df['Volume_ZScore'] = (df['Volume'] - df['Volume_MA']) / df['Volume_StdDev']
        
        return df
    
    def resample_data(self,
                     df: pd.DataFrame,
                     timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original data
        timeframe : str
            New timeframe ('1H', '4H', '1D', etc.)
            
        Returns:
        --------
        pd.DataFrame
            Resampled data
        """
        resampled = pd.DataFrame()
        
        # Resample OHLCV
        resampled['Open'] = df['Open'].resample(timeframe).first()
        resampled['High'] = df['High'].resample(timeframe).max()
        resampled['Low'] = df['Low'].resample(timeframe).min()
        resampled['Close'] = df['Close'].resample(timeframe).last()
        resampled['Volume'] = df['Volume'].resample(timeframe).sum()
        
        # Handle missing values
        resampled = self._handle_missing_values(resampled)
        
        return resampled
    
    def add_technical_indicators(self,
                               df: pd.DataFrame,
                               indicators: List[Dict]) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        indicators : List[Dict]
            List of indicator configurations
            
        Returns:
        --------
        pd.DataFrame
            Data with added indicators
        """
        for indicator in indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            
            if name == 'SMA':
                df[f"SMA_{params['period']}"] = df['Close'].rolling(
                    window=params['period']).mean()
                    
            elif name == 'EMA':
                df[f"EMA_{params['period']}"] = df['Close'].ewm(
                    span=params['period']).mean()
                    
            elif name == 'RSI':
                period = params.get('period', 14)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
                
            elif name == 'BB':
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2)
                ma = df['Close'].rolling(window=period).mean()
                std = df['Close'].rolling(window=period).std()
                df[f'BB_Upper_{period}'] = ma + (std * std_dev)
                df[f'BB_Lower_{period}'] = ma - (std * std_dev)
                df[f'BB_Middle_{period}'] = ma
        
        return df
    
    def merge_data(self,
                  dfs: List[pd.DataFrame],
                  on: str = 'Close',
                  suffixes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge multiple dataframes
        
        Parameters:
        -----------
        dfs : List[pd.DataFrame]
            List of dataframes to merge
        on : str
            Column to merge on
        suffixes : List[str], optional
            Suffixes for merged columns
            
        Returns:
        --------
        pd.DataFrame
            Merged dataframe
        """
        if not suffixes:
            suffixes = [f'_{i}' for i in range(len(dfs))]
            
        result = dfs[0]
        for i, df in enumerate(dfs[1:], 1):
            result = pd.merge(
                result,
                df,
                left_index=True,
                right_index=True,
                suffixes=(suffixes[i-1], suffixes[i])
            )
            
        return result
    
    @staticmethod
    def calculate_returns(prices: pd.Series,
                         method: str = 'arithmetic') -> pd.Series:
        """
        Calculate returns from price series
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        method : str
            Return calculation method ('arithmetic' or 'log')
            
        Returns:
        --------
        pd.Series
            Returns series
        """
        if method == 'arithmetic':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'arithmetic' or 'log'")
    
    def prepare_features(self,
                        df: pd.DataFrame,
                        feature_config: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for analysis or modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        feature_config : Dict
            Feature configuration
            
        Returns:
        --------
        Tuple[pd.DataFrame, List[str]]
            Processed features and feature names
        """
        features = pd.DataFrame(index=df.index)
        feature_names = []
        
        for feature in feature_config:
            if feature['type'] == 'price':
                features[feature['name']] = df[feature['column']]
                feature_names.append(feature['name'])
                
            elif feature['type'] == 'indicator':
                df = self.add_technical_indicators(df, [feature])
                features[feature['name']] = df[feature['name']]
                feature_names.append(feature['name'])
                
            elif feature['type'] == 'return':
                features[feature['name']] = self.calculate_returns(
                    df[feature['column']], feature.get('method', 'arithmetic')
                )
                feature_names.append(feature['name'])
        
        return features, feature_names

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Load and process data
    data = processor.load_data(
        symbol="EURUSD=X",
        start_date="2023-01-01",
        end_date="2024-01-03"
    )
    
    # Add technical indicators
    indicators = [
        {'name': 'SMA', 'params': {'period': 20}},
        {'name': 'RSI', 'params': {'period': 14}},
        {'name': 'BB', 'params': {'period': 20, 'std_dev': 2}}
    ]
    
    data = processor.add_technical_indicators(data, indicators)
    
    print("\nData Overview:")
    print(data.tail())