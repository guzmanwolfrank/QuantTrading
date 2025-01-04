# strategies/bollinger_bounce.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class BollingerBounceStrategy:
    """
    Bollinger Bands Bounce Trading Strategy
    
    This strategy generates signals based on price bounces from Bollinger Bands
    combined with band contraction (squeeze) and momentum confirmation.
    
    Parameters:
    -----------
    bb_period : int
        Period for Bollinger Bands calculation (default: 20)
    bb_std : float
        Number of standard deviations for bands (default: 2.0)
    squeeze_factor : float
        Threshold for identifying band squeeze (default: 0.8)
    rsi_period : int
        Period for RSI calculation (default: 14)
    rsi_threshold : Tuple[float, float]
        Thresholds for oversold and overbought (default: (30, 70))
    exit_bars : int
        Maximum number of bars to hold position (default: 5)
    """
    
    def __init__(self, 
                 bb_period: int = 20, 
                 bb_std: float = 2.0,
                 squeeze_factor: float = 0.8,
                 rsi_period: int = 14,
                 rsi_threshold: Tuple[float, float] = (30, 70),
                 exit_bars: int = 5):
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_factor = squeeze_factor
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.exit_bars = exit_bars
        self.name = "Bollinger Bounce"
        
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy indicators
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added indicator columns
        """
        # Calculate Bollinger Bands
        df['MA'] = df['Close'].rolling(window=self.bb_period).mean()
        rolling_std = df['Close'].rolling(window=self.bb_period).std()
        
        df['BB_upper'] = df['MA'] + (rolling_std * self.bb_std)
        df['BB_lower'] = df['MA'] - (rolling_std * self.bb_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA']
        
        # Calculate band squeeze
        df['BB_squeeze'] = df['BB_width'] < (df['BB_width'].rolling(window=20).mean() * self.squeeze_factor)
        
        # Calculate RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Calculate price position relative to bands
        df['Price_Position'] = (df['Close'] - df['MA']) / (df['BB_upper'] - df['MA'])
        
        # Momentum indicators
        df['Momentum'] = df['Close'].pct_change(periods=self.bb_period)
        df['Volume_Factor'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added signal columns
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signals
        df['Signal'] = 0
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        # Long signal conditions
        long_conditions = (
            (df['Close'] <= df['BB_lower']) &  # Price at lower band
            (df['BB_squeeze']) &                # Bands are squeezed
            (df['RSI'] < self.rsi_threshold[0]) &  # Oversold
            (df['Volume_Factor'] > 1.0)         # Above average volume
        )
        
        # Short signal conditions
        short_conditions = (
            (df['Close'] >= df['BB_upper']) &   # Price at upper band
            (df['BB_squeeze']) &                 # Bands are squeezed
            (df['RSI'] > self.rsi_threshold[1]) &  # Overbought
            (df['Volume_Factor'] > 1.0)          # Above average volume
        )
        
        # Set signals
        df.loc[long_conditions, 'Signal'] = 1
        df.loc[short_conditions, 'Signal'] = -1
        
        # Calculate stop loss and take profit levels
        band_width = df['BB_upper'] - df['BB_lower']
        
        # For long positions
        df.loc[long_conditions, 'Entry_Price'] = df['Close']
        df.loc[long_conditions, 'Stop_Loss'] = df['Close'] - (band_width * 0.3)
        df.loc[long_conditions, 'Take_Profit'] = df['Close'] + (band_width * 0.5)
        
        # For short positions
        df.loc[short_conditions, 'Entry_Price'] = df['Close']
        df.loc[short_conditions, 'Stop_Loss'] = df['Close'] + (band_width * 0.3)
        df.loc[short_conditions, 'Take_Profit'] = df['Close'] - (band_width * 0.5)
        
        # Add position management
        self._apply_position_management(df)
        
        return df
    
    def _apply_position_management(self, df: pd.DataFrame) -> None:
        """Apply position management rules"""
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        bars_held = 0
        
        for i in range(1, len(df)):
            # Check for new signals
            if df['Signal'].iloc[i] != 0:
                position = df['Signal'].iloc[i]
                entry_price = df['Entry_Price'].iloc[i]
                stop_loss = df['Stop_Loss'].iloc[i]
                take_profit = df['Take_Profit'].iloc[i]
                bars_held = 0
                continue
                
            # Update position tracking
            if position != 0:
                bars_held += 1
                current_price = df['Close'].iloc[i]
                
                # Check stop loss
                if (position == 1 and current_price < stop_loss) or \
                   (position == -1 and current_price > stop_loss):
                    df.loc[df.index[i], 'Signal'] = -position
                    position = 0
                    bars_held = 0
                    
                # Check take profit
                elif (position == 1 and current_price > take_profit) or \
                     (position == -1 and current_price < take_profit):
                    df.loc[df.index[i], 'Signal'] = -position
                    position = 0
                    bars_held = 0
                    
                # Check maximum bars held
                elif bars_held >= self.exit_bars:
                    df.loc[df.index[i], 'Signal'] = -position
                    position = 0
                    bars_held = 0
    
    def optimize_parameters(self, 
                          df: pd.DataFrame, 
                          param_ranges: Optional[Dict] = None) -> Dict:
        """
        Optimize strategy parameters
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data for optimization
        param_ranges : Dict, optional
            Ranges for parameter optimization
            
        Returns:
        --------
        Dict
            Optimized parameters and performance metrics
        """
        if param_ranges is None:
            param_ranges = {
                'bb_period': range(10, 31, 5),
                'bb_std': [1.5, 2.0, 2.5, 3.0],
                'squeeze_factor': [0.6, 0.7, 0.8, 0.9]
            }
            
        best_sharpe = -np.inf
        best_params = {}
        results = []
        
        # Grid search
        for bb_period in param_ranges['bb_period']:
            for bb_std in param_ranges['bb_std']:
                for squeeze in param_ranges['squeeze_factor']:
                    # Set parameters
                    self.bb_period = bb_period
                    self.bb_std = bb_std
                    self.squeeze_factor = squeeze
                    
                    # Run backtest
                    test_df = self.generate_signals(df.copy())
                    
                    # Calculate returns
                    test_df['Returns'] = test_df['Signal'].shift(1) * test_df['Close'].pct_change()
                    sharpe = np.sqrt(252) * test_df['Returns'].mean() / test_df['Returns'].std()
                    
                    results.append({
                        'bb_period': bb_period,
                        'bb_std': bb_std,
                        'squeeze_factor': squeeze,
                        'sharpe_ratio': sharpe
                    })
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            'bb_period': bb_period,
                            'bb_std': bb_std,
                            'squeeze_factor': squeeze
                        }
        
        return {
            'best_parameters': best_params,
            'best_sharpe': best_sharpe,
            'all_results': pd.DataFrame(results)
        }

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    symbol = "EURUSD=X"
    data = yf.download(symbol, start="2023-01-01", end="2024-01-03")
    
    # Create and run strategy
    strategy = BollingerBounceStrategy()
    results = strategy.generate_signals(data)
    
    # Print sample results
    print("\nStrategy Signals:")
    print(results[['Close', 'Signal', 'BB_upper', 'BB_lower', 'RSI']].tail())