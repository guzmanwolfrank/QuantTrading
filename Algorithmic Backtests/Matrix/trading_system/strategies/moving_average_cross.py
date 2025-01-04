# strategies/moving_average_cross.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class MovingAverageCrossStrategy:
    """
    Moving Average Crossover Trading Strategy.
    Generates signals based on fast and slow moving average crossovers.
    
    Parameters:
    -----------
    fast_period : int
        Period for fast moving average (default: 10)
    slow_period : int
        Period for slow moving average (default: 30)
    exit_bars : int
        Maximum number of bars to hold position (default: 5)
    trend_filter : bool
        Use trend filter for signal confirmation (default: True)
    """
    
    def __init__(self,
                 fast_period: int = 10,
                 slow_period: int = 30,
                 exit_bars: int = 5,
                 trend_filter: bool = True):
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.exit_bars = exit_bars
        self.trend_filter = trend_filter
        self.name = "Moving Average Cross"
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy indicators"""
        # Calculate moving averages
        df['FastMA'] = df['Close'].rolling(window=self.fast_period).mean()
        df['SlowMA'] = df['Close'].rolling(window=self.slow_period).mean()
        
        # Calculate crossover signals
        df['MA_Spread'] = df['FastMA'] - df['SlowMA']
        df['Cross_Up'] = (df['MA_Spread'] > 0) & (df['MA_Spread'].shift(1) <= 0)
        df['Cross_Down'] = (df['MA_Spread'] < 0) & (df['MA_Spread'].shift(1) >= 0)
        
        if self.trend_filter:
            # Add trend filter using longer MA
            df['Trend_MA'] = df['Close'].rolling(window=self.slow_period * 2).mean()
            df['Trend_Up'] = df['Close'] > df['Trend_MA']
            df['Trend_Down'] = df['Close'] < df['Trend_MA']
        
        # Calculate momentum and volatility indicators
        df['Momentum'] = df['Close'].pct_change(periods=self.fast_period)
        df['Volatility'] = df['Close'].pct_change().rolling(window=self.slow_period).std()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(df)
        
        # Initialize signals
        df['Signal'] = 0
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        # Generate signals with trend filter
        if self.trend_filter:
            long_condition = df['Cross_Up'] & df['Trend_Up']
            short_condition = df['Cross_Down'] & df['Trend_Down']
        else:
            long_condition = df['Cross_Up']
            short_condition = df['Cross_Down']
        
        # Apply volatility-based position sizing
        volatility_factor = 1 / df['Volatility'].rolling(window=self.slow_period).mean()
        volatility_factor = volatility_factor / volatility_factor.mean()
        
        # Set signals
        df.loc[long_condition, 'Signal'] = 1 * volatility_factor[long_condition]
        df.loc[short_condition, 'Signal'] = -1 * volatility_factor[short_condition]
        
        # Set stop loss and take profit levels
        atr_multiple = 2
        atr = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
        
        # For long positions
        df.loc[long_condition, 'Entry_Price'] = df['Close']
        df.loc[long_condition, 'Stop_Loss'] = df['Close'] - (atr * atr_multiple)
        df.loc[long_condition, 'Take_Profit'] = df['Close'] + (atr * atr_multiple * 1.5)
        
        # For short positions
        df.loc[short_condition, 'Entry_Price'] = df['Close']
        df.loc[short_condition, 'Stop_Loss'] = df['Close'] + (atr * atr_multiple)
        df.loc[short_condition, 'Take_Profit'] = df['Close'] - (atr * atr_multiple * 1.5)
        
        # Apply position management
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
                if (position > 0 and current_price < stop_loss) or \
                   (position < 0 and current_price > stop_loss):
                    df.loc[df.index[i], 'Signal'] = -position
                    position = 0
                    bars_held = 0
                
                # Check take profit
                elif (position > 0 and current_price > take_profit) or \
                     (position < 0 and current_price < take_profit):
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
        """Optimize strategy parameters"""
        if param_ranges is None:
            param_ranges = {
                'fast_period': range(5, 21, 5),
                'slow_period': range(20, 41, 5)
            }
        
        best_sharpe = -np.inf
        best_params = {}
        results = []
        
        for fast_period in param_ranges['fast_period']:
            for slow_period in param_ranges['slow_period']:
                if fast_period >= slow_period:
                    continue
                
                # Test parameters
                self.fast_period = fast_period
                self.slow_period = slow_period
                
                # Run backtest
                test_df = self.generate_signals(df.copy())
                test_df['Returns'] = test_df['Signal'].shift(1) * test_df['Close'].pct_change()
                
                # Calculate Sharpe ratio
                sharpe = np.sqrt(252) * test_df['Returns'].mean() / test_df['Returns'].std() \
                    if test_df['Returns'].std() != 0 else 0
                
                results.append({
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'sharpe_ratio': sharpe
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'fast_period': fast_period,
                        'slow_period': slow_period
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
    data = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-03")
    
    # Create and run strategy
    strategy = MovingAverageCrossStrategy()
    results = strategy.generate_signals(data)
    
    # Print sample results
    print("\nStrategy Signals:")
    print(results[['Close', 'Signal', 'FastMA', 'SlowMA']].tail())