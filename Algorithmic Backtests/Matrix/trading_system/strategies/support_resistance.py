# strategies/support_resistance.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema

class SupportResistanceStrategy:
    """
    Support and Resistance Trading Strategy
    
    Identifies key support and resistance levels using various methods:
    - Local minima and maxima
    - Volume-weighted price clusters
    - Historical price pivots
    
    Parameters:
    -----------
    window : int
        Lookback window for level identification (default: 20)
    threshold : float
        Minimum distance between levels (default: 0.02)
    volume_factor : float
        Minimum volume factor for level confirmation (default: 1.5)
    touch_count : int
        Minimum touches required to confirm level (default: 3)
    exit_bars : int
        Maximum bars to hold position (default: 10)
    atr_periods : int
        Periods for ATR calculation (default: 14)
    """
    
    def __init__(self,
                 window: int = 20,
                 threshold: float = 0.02,
                 volume_factor: float = 1.5,
                 touch_count: int = 3,
                 exit_bars: int = 10,
                 atr_periods: int = 14):
        
        self.window = window
        self.threshold = threshold
        self.volume_factor = volume_factor
        self.touch_count = touch_count
        self.exit_bars = exit_bars
        self.atr_periods = atr_periods
        self.name = "Support Resistance"
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=self.atr_periods).mean()
    
    def find_extrema(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find local minima and maxima
        
        Returns:
        --------
        Tuple[List[float], List[float]]
            Support and resistance levels
        """
        # Find local minima
        local_min_idx = argrelextrema(df['Low'].values, np.less_all, order=self.window)[0]
        support_levels = df['Low'].iloc[local_min_idx].values
        
        # Find local maxima
        local_max_idx = argrelextrema(df['High'].values, np.greater_all, order=self.window)[0]
        resistance_levels = df['High'].iloc[local_max_idx].values
        
        return support_levels.tolist(), resistance_levels.tolist()
    
    def find_volume_clusters(self, df: pd.DataFrame) -> List[float]:
        """
        Find price levels with high volume concentration
        
        Returns:
        --------
        List[float]
            Price levels with significant volume
        """
        # Calculate price-volume distribution
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 100)
        volume_profile = np.zeros_like(price_bins)
        
        for i in range(len(df)):
            idx_range = (price_bins >= df['Low'].iloc[i]) & (price_bins <= df['High'].iloc[i])
            volume_profile[idx_range] += df['Volume'].iloc[i]
            
        # Find peaks in volume profile
        peak_idx = argrelextrema(volume_profile, np.greater)[0]
        significant_levels = price_bins[peak_idx][volume_profile[peak_idx] > 
                                                volume_profile.mean() * self.volume_factor]
        
        return significant_levels.tolist()
    
    def validate_level(self, price: float, level: float, atr: float) -> bool:
        """Check if price is near a level"""
        return abs(price - level) < atr * 0.5
    
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
        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Find support and resistance levels
        support_levels, resistance_levels = self.find_extrema(df)
        volume_levels = self.find_volume_clusters(df)
        
        # Combine and filter levels
        all_levels = sorted(set(support_levels + resistance_levels + volume_levels))
        
        # Initialize level tracking
        df['Nearest_Support'] = np.nan
        df['Nearest_Resistance'] = np.nan
        df['Distance_To_Support'] = np.nan
        df['Distance_To_Resistance'] = np.nan
        
        # Track touches for each level
        level_touches = {level: 0 for level in all_levels}
        
        # Find nearest levels for each bar
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            valid_levels = [level for level in all_levels if 
                          abs(price - level) < df['ATR'].iloc[i] * 2]
            
            if valid_levels:
                below_levels = [l for l in valid_levels if l < price]
                above_levels = [l for l in valid_levels if l > price]
                
                if below_levels:
                    nearest_support = max(below_levels)
                    df.loc[df.index[i], 'Nearest_Support'] = nearest_support
                    df.loc[df.index[i], 'Distance_To_Support'] = price - nearest_support
                    
                if above_levels:
                    nearest_resistance = min(above_levels)
                    df.loc[df.index[i], 'Nearest_Resistance'] = nearest_resistance
                    df.loc[df.index[i], 'Distance_To_Resistance'] = nearest_resistance - price
                    
                # Update level touches
                for level in valid_levels:
                    if self.validate_level(price, level, df['ATR'].iloc[i]):
                        level_touches[level] += 1
        
        # Filter for confirmed levels
        confirmed_levels = [level for level, touches in level_touches.items() 
                          if touches >= self.touch_count]
        
        df['Confirmed_Levels'] = str(confirmed_levels)
        
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
        
        # Generate signals based on level bounces
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            atr = df['ATR'].iloc[i]
            
            # Long signal conditions
            if (df['Distance_To_Support'].iloc[i-1] is not None and 
                df['Distance_To_Support'].iloc[i-1] < atr * 0.5 and 
                price > prev_price):
                
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Entry_Price'] = price
                df.loc[df.index[i], 'Stop_Loss'] = price - atr
                df.loc[df.index[i], 'Take_Profit'] = price + (atr * 2)
            
            # Short signal conditions
            elif (df['Distance_To_Resistance'].iloc[i-1] is not None and 
                  df['Distance_To_Resistance'].iloc[i-1] < atr * 0.5 and 
                  price < prev_price):
                
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Entry_Price'] = price
                df.loc[df.index[i], 'Stop_Loss'] = price + atr
                df.loc[df.index[i], 'Take_Profit'] = price - (atr * 2)
        
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
                'window': range(10, 31, 5),
                'threshold': [0.01, 0.02, 0.03, 0.04],
                'touch_count': [2, 3, 4, 5]
            }
        
        best_sharpe = -np.inf
        best_params = {}
        results = []
        
        # Grid search
        for window in param_ranges['window']:
            for threshold in param_ranges['threshold']:
                for touch_count in param_ranges['touch_count']:
                    # Set parameters
                    self.window = window
                    self.threshold = threshold
                    self.touch_count = touch_count
                    
                    # Run backtest
                    test_df = self.generate_signals(df.copy())
                    
                    # Calculate returns
                    test_df['Returns'] = test_df['Signal'].shift(1) * test_df['Close'].pct_change()
                    sharpe = np.sqrt(252) * test_df['Returns'].mean() / test_df['Returns'].std()
                    
                    results.append({
                        'window': window,
                        'threshold': threshold,
                        'touch_count': touch_count,
                        'sharpe_ratio': sharpe
                    })
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            'window': window,
                            'threshold': threshold,
                            'touch_count': touch_count
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
    strategy = SupportResistanceStrategy()
    results = strategy.generate_signals(data)
    
    # Print sample results
    print("\nStrategy Signals:")
    print(results[['Close', 'Signal', 'Nearest_Support', 
                  'Nearest_Resistance', 'ATR']].tail())