# utils/risk_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class RiskManager:
    """
    Risk Management System for Trading Strategies.
    
    Handles:
    - Position sizing
    - Risk per trade
    - Portfolio exposure
    - Stop loss management
    - Correlation management
    - Drawdown control
    
    Parameters:
    -----------
    max_position_size : float
        Maximum position size as fraction of portfolio (default: 0.02)
    max_portfolio_risk : float
        Maximum portfolio risk as fraction of portfolio (default: 0.02)
    max_correlation : float
        Maximum correlation between positions (default: 0.7)
    max_drawdown : float
        Maximum allowed drawdown before reducing exposure (default: 0.20)
    risk_free_rate : float
        Annual risk-free rate (default: 0.02)
    """
    
    def __init__(self,
                 max_position_size: float = 0.02,
                 max_portfolio_risk: float = 0.02,
                 max_correlation: float = 0.7,
                 max_drawdown: float = 0.20,
                 risk_free_rate: float = 0.02):
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        
    def calculate_position_size(self,
                              capital: float,
                              price: float,
                              volatility: float,
                              stop_loss: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters
        
        Parameters:
        -----------
        capital : float
            Available trading capital
        price : float
            Current asset price
        volatility : float
            Asset volatility (standard deviation)
        stop_loss : float, optional
            Stop loss price level
            
        Returns:
        --------
        float
            Recommended position size in units
        """
        # Calculate risk amount
        risk_amount = capital * self.max_position_size
        
        if stop_loss:
            # Position size based on stop loss
            risk_per_unit = abs(price - stop_loss)
            position_size = risk_amount / risk_per_unit
        else:
            # Position size based on volatility
            risk_per_unit = price * volatility
            position_size = risk_amount / risk_per_unit
        
        # Ensure position size doesn't exceed max portfolio risk
        max_size = (capital * self.max_portfolio_risk) / price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def apply_position_sizing(self, 
                            df: pd.DataFrame,
                            initial_capital: float = 100000,
                            volatility_window: int = 20) -> pd.DataFrame:
        """
        Apply position sizing to trading signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with signals
        initial_capital : float
            Initial trading capital
        volatility_window : int
            Window for volatility calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with position sizes
        """
        df = df.copy()
        
        # Calculate volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=volatility_window).std()
        
        # Initialize position tracking
        df['Position_Size'] = 0.0
        df['Capital'] = initial_capital
        current_position = 0
        
        for i in range(1, len(df)):
            # Update capital based on previous returns
            if current_position != 0:
                pnl = current_position * df['Close'].iloc[i] * \
                      (df['Close'].iloc[i] / df['Close'].iloc[i-1] - 1)
                df.loc[df.index[i], 'Capital'] = df['Capital'].iloc[i-1] + pnl
            else:
                df.loc[df.index[i], 'Capital'] = df['Capital'].iloc[i-1]
            
            # Check for new signal
            if df['Signal'].iloc[i] != 0:
                # Calculate position size
                size = self.calculate_position_size(
                    capital=df['Capital'].iloc[i],
                    price=df['Close'].iloc[i],
                    volatility=df['Volatility'].iloc[i],
                    stop_loss=df['Stop_Loss'].iloc[i] if 'Stop_Loss' in df.columns else None
                )
                
                df.loc[df.index[i], 'Position_Size'] = size * df['Signal'].iloc[i]
                current_position = size * df['Signal'].iloc[i]
            else:
                df.loc[df.index[i], 'Position_Size'] = current_position
        
        return df
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics for the strategy
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with positions
            
        Returns:
        --------
        Dict
            Dictionary of risk metrics
        """
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Position_Size'].shift(1) * df['Returns']
        
        # Calculate drawdown
        cum_returns = (1 + df['Strategy_Returns']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = cum_returns / rolling_max - 1
        
        # Calculate risk metrics
        volatility = df['Strategy_Returns'].std() * np.sqrt(252)
        var_95 = np.percentile(df['Strategy_Returns'], 5)
        cvar_95 = df['Strategy_Returns'][df['Strategy_Returns'] <= var_95].mean()
        
        return {
            'Current_Drawdown': drawdown.iloc[-1],
            'Max_Drawdown': drawdown.min(),
            'Volatility': volatility,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Avg_Position_Size': df['Position_Size'].abs().mean(),
            'Max_Position_Size': df['Position_Size'].abs().max(),
            'Current_Risk_Exposure': (df['Position_Size'].iloc[-1] * 
                                    df['Volatility'].iloc[-1])
        }
    
    def check_correlation_limits(self,
                               returns: pd.DataFrame,
                               new_position: str) -> bool:
        """
        Check if adding new position violates correlation limits
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns of current positions
        new_position : str
            Symbol of new position to add
            
        Returns:
        --------
        bool
            True if correlation limits are satisfied
        """
        if new_position in returns.columns:
            corr_matrix = returns.corr()
            max_corr = corr_matrix[new_position].abs().max()
            return max_corr <= self.max_correlation
        return True
    
    def adjust_for_drawdown(self,
                           current_drawdown: float,
                           position_size: float) -> float:
        """
        Adjust position size based on drawdown
        
        Parameters:
        -----------
        current_drawdown : float
            Current drawdown level
        position_size : float
            Calculated position size
            
        Returns:
        --------
        float
            Adjusted position size
        """
        if abs(current_drawdown) > self.max_drawdown:
            # Reduce position size proportionally to drawdown excess
            reduction_factor = 1 - (abs(current_drawdown) - self.max_drawdown)
            return position_size * max(0, reduction_factor)
        return position_size
    
    def calculate_kelly_criterion(self,
                                win_rate: float,
                                win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion for position sizing
        
        Parameters:
        -----------
        win_rate : float
            Probability of winning trade
        win_loss_ratio : float
            Ratio of average win to average loss
            
        Returns:
        --------
        float
            Kelly Criterion value
        """
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Fractional Kelly for more conservative sizing
        return max(0, kelly * 0.5)
    
    def generate_risk_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with positions
            
        Returns:
        --------
        Dict
            Risk report metrics
        """
        metrics = self.calculate_risk_metrics(df)
        
        # Calculate additional risk measures
        daily_returns = df['Strategy_Returns'].dropna()
        rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
        
        report = {
            **metrics,
            'Current_Volatility': rolling_vol.iloc[-1],
            'Risk_Per_Trade': (df['Position_Size'].abs() * df['Volatility']).mean(),
            'Exposure_Ratio': df['Position_Size'].abs().mean() / \
                            df['Position_Size'].abs().max(),
            'Time_In_Market': len(df[df['Position_Size'] != 0]) / len(df)
        }
        
        return report

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    data = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-03")
    
    # Add sample signals
    data['Signal'] = np.random.choice([-1, 0, 1], size=len(data))
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_position_size=0.02,
        max_portfolio_risk=0.02
    )
    
    # Apply position sizing
    results = risk_manager.apply_position_sizing(data)
    
    # Generate risk report
    risk_report = risk_manager.generate_risk_report(results)
    
    # Print results
    print("\nRisk Report:")
    for metric, value in risk_report.items():
        print(f"{metric}: {value:.4f}")