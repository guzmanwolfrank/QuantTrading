# utils/performance_metrics.py

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """
    Trading strategy performance analysis.
    Calculates comprehensive trading metrics, risk measures, and statistics.
    
    Parameters:
    -----------
    risk_free_rate : float
        Annual risk-free rate for ratio calculations (default: 0.02)
    trading_days : int
        Number of trading days per year (default: 252)
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_risk_free = (1 + risk_free_rate) ** (1/trading_days) - 1
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with signals and returns
            
        Returns:
        --------
        Dict
            Dictionary of performance metrics
        """
        # Ensure required columns exist
        required_cols = ['Close', 'Signal', 'Returns']
        if not all(col in df.columns for col in required_cols):
            df['Returns'] = df['Close'].pct_change()
            
        # Calculate strategy returns
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(df)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(df)
        
        # Return distribution metrics
        distribution_metrics = self._calculate_distribution_metrics(df)
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(df)
        
        # Trade timing metrics
        timing_metrics = self._calculate_timing_metrics(df)
        
        # Combine all metrics
        all_metrics = {
            **trading_metrics,
            **risk_metrics,
            **distribution_metrics,
            **drawdown_metrics,
            **timing_metrics
        }
        
        return all_metrics
    
    def _calculate_trading_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate trading performance metrics"""
        trades = df[df['Signal'] != 0].copy()
        strategy_returns = df['Strategy_Returns'].dropna()
        
        total_trades = len(trades)
        winning_trades = len(strategy_returns[strategy_returns > 0])
        losing_trades = len(strategy_returns[strategy_returns < 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = strategy_returns[strategy_returns > 0].mean() if winning_trades > 0 else 0
            avg_loss = strategy_returns[strategy_returns < 0].mean() if losing_trades > 0 else 0
            profit_factor = abs(strategy_returns[strategy_returns > 0].sum() / 
                              strategy_returns[strategy_returns < 0].sum()) if losing_trades > 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            
        return {
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate': win_rate,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Profit_Factor': profit_factor,
            'Total_Return': (1 + df['Strategy_Returns']).prod() - 1,
            'Annual_Return': (1 + df['Strategy_Returns']).prod() ** (self.trading_days/len(df)) - 1
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk and risk-adjusted return metrics"""
        returns = df['Strategy_Returns'].dropna()
        
        # Volatility
        volatility = returns.std() * np.sqrt(self.trading_days)
        
        # Sharpe Ratio
        excess_returns = returns - self.daily_risk_free
        sharpe = np.sqrt(self.trading_days) * excess_returns.mean() / returns.std() \
            if returns.std() != 0 else 0
            
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(self.trading_days) * returns.mean() / downside_returns.std() \
            if len(downside_returns) > 0 else 0
            
        # Calmar Ratio
        max_dd = self._calculate_drawdown_metrics(df)['Max_Drawdown']
        calmar = abs(returns.mean() * self.trading_days / max_dd) if max_dd != 0 else 0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Information Ratio
        benchmark_returns = df['Returns'].dropna()
        active_returns = returns - benchmark_returns
        info_ratio = np.sqrt(self.trading_days) * active_returns.mean() / active_returns.std() \
            if len(active_returns) > 0 and active_returns.std() != 0 else 0
        
        return {
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar,
            'Information_Ratio': info_ratio,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'VaR_99': var_99,
            'CVaR_99': cvar_99
        }
    
    def _calculate_distribution_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate return distribution metrics"""
        returns = df['Strategy_Returns'].dropna()
        
        return {
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Daily_Std': returns.std(),
            'Daily_Mean': returns.mean(),
            'Best_Day': returns.max(),
            'Worst_Day': returns.min()
        }
    
    def _calculate_drawdown_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate drawdown metrics"""
        cum_returns = (1 + df['Strategy_Returns']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        
        # Calculate drawdown periods
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown.ne(is_drawdown.shift()).cumsum()
        drawdown_periods = drawdowns.groupby(drawdown_starts).agg(['min', 'count'])
        
        return {
            'Max_Drawdown': drawdowns.min(),
            'Avg_Drawdown': drawdowns[drawdowns < 0].mean(),
            'Avg_Drawdown_Days': drawdown_periods['count'].mean(),
            'Max_Drawdown_Days': drawdown_periods['count'].max(),
            'Current_Drawdown': drawdowns.iloc[-1]
        }
    
    def _calculate_timing_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate market timing metrics"""
        # Calculate up/down market capture
        market_returns = df['Returns'].dropna()
        strategy_returns = df['Strategy_Returns'].dropna()
        
        up_market = market_returns > 0
        down_market = market_returns < 0
        
        up_capture = strategy_returns[up_market].mean() / market_returns[up_market].mean() \
            if len(market_returns[up_market]) > 0 else 0
        down_capture = strategy_returns[down_market].mean() / market_returns[down_market].mean() \
            if len(market_returns[down_market]) > 0 else 0
            
        # Calculate trade duration statistics
        trades = df[df['Signal'] != 0].copy()
        if len(trades) > 1:
            trade_durations = trades.index.to_series().diff().dt.total_seconds() / (24 * 3600)
            avg_duration = trade_durations.mean()
            max_duration = trade_durations.max()
        else:
            avg_duration = max_duration = 0
            
        return {
            'Up_Market_Capture': up_capture,
            'Down_Market_Capture': down_capture,
            'Avg_Trade_Duration_Days': avg_duration,
            'Max_Trade_Duration_Days': max_duration
        }
    
    def generate_trade_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed trade list
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with signals
            
        Returns:
        --------
        pd.DataFrame
            Detailed trade information
        """
        trades = []
        current_position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(df)):
            if df['Signal'].iloc[i] != 0:
                if current_position == 0:  # Opening trade
                    current_position = df['Signal'].iloc[i]
                    entry_price = df['Close'].iloc[i]
                    entry_time = df.index[i]
                else:  # Closing trade
                    exit_price = df['Close'].iloc[i]
                    returns = (exit_price - entry_price) / entry_price * current_position
                    
                    trades.append({
                        'Entry_Time': entry_time,
                        'Exit_Time': df.index[i],
                        'Direction': 'Long' if current_position == 1 else 'Short',
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Returns': returns,
                        'Duration': (df.index[i] - entry_time).total_seconds() / (24 * 3600)
                    })
                    
                    current_position = 0
        
        return pd.DataFrame(trades)

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    data = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-03")
    
    # Add sample signals (for demonstration)
    data['Signal'] = np.random.choice([-1, 0, 1], size=len(data))
    
    # Calculate metrics
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(data)
    
    # Print results
    print("\nPerformance Metrics:")
    for category, value in metrics.items():
        if isinstance(value, float):
            print(f"{category}: {value:.4f}")
        else:
            print(f"{category}: {value}")
            
    # Generate trade list
    trades = analyzer.generate_trade_list(data)
    print("\nSample Trades:")
    print(trades.head())