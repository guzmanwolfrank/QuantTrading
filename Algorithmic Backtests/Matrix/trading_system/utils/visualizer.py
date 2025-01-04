# utils/visualizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import mplfinance as mpf

class Visualizer:
    """
    Trading strategy visualization toolkit.
    Creates various plots and charts for strategy analysis.
    
    Parameters:
    -----------
    style : str
        Plot style ('seaborn', 'dark_background', etc.)
    default_figsize : tuple
        Default figure size
    """
    
    def __init__(self, 
                 style: str = 'seaborn',
                 default_figsize: Tuple[int, int] = (12, 8)):
        
        self.style = style
        self.default_figsize = default_figsize
        plt.style.use(style)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    
    def create_strategy_dashboard(self,
                                df: pd.DataFrame,
                                metrics: Dict,
                                filename: Optional[str] = None) -> None:
        """
        Create comprehensive strategy dashboard using plotly
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with signals and returns
        metrics : Dict
            Performance metrics
        filename : str, optional
            File to save the dashboard
        """
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price and Signals',
                'Equity Curve',
                'Returns Distribution',
                'Drawdown',
                'Monthly Returns',
                'Rolling Metrics'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Price and Signals
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Add buy/sell signals
        buys = df[df['Signal'] == 1]
        sells = df[df['Signal'] == -1]
        
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys['Close'],
                mode='markers',
                name='Buy',
                marker=dict(color=self.colors['positive'], size=10)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells['Close'],
                mode='markers',
                name='Sell',
                marker=dict(color=self.colors['negative'], size=10)
            ),
            row=1, col=1
        )
        
        # 2. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=(1 + df['Strategy_Returns']).cumprod(),
                name='Equity',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=2
        )
        
        # 3. Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=df['Strategy_Returns'],
                name='Returns',
                nbinsx=50,
                marker_color=self.colors['primary']
            ),
            row=2, col=1
        )
        
        # 4. Drawdown
        drawdown = (1 + df['Strategy_Returns']).cumprod() / \
                  (1 + df['Strategy_Returns']).cumprod().expanding().max() - 1
                  
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=self.colors['negative'])
            ),
            row=2, col=2
        )
        
        # 5. Monthly Returns Heatmap
        monthly_returns = df['Strategy_Returns'].resample('M').sum().to_frame()
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        pivot = monthly_returns.pivot(
            index='Year',
            columns='Month',
            values='Strategy_Returns'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn'
            ),
            row=3, col=1
        )
        
        # 6. Rolling Metrics
        window = 252  # One year
        rolling_sharpe = np.sqrt(252) * df['Strategy_Returns'].rolling(window).mean() / \
                        df['Strategy_Returns'].rolling(window).std()
                        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color=self.colors['primary'])
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="Strategy Analysis Dashboard"
        )
        
        # Add metrics annotations
        metrics_text = f"""
        Total Return: {metrics['Total_Return']:.2%}
        Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}
        Max Drawdown: {metrics['Max_Drawdown']:.2%}
        Win Rate: {metrics['Win_Rate']:.2%}
        """
        
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.0,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
        
        if filename:
            fig.write_html(filename)
            
        fig.show()
    
    def plot_candlestick(self,
                        df: pd.DataFrame,
                        signals: bool = True,
                        volume: bool = True) -> None:
        """
        Create candlestick chart with signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with signals
        signals : bool
            Whether to plot signals
        volume : bool
            Whether to plot volume
        """
        # Prepare the data
        data = df.copy()
        
        # Create custom style
        mc = mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick='inherit',
            volume='in',
            ohlc='inherit'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='dotted',
            figcolor='white'
        )
        
        # Prepare plot kwargs
        kwargs = {
            'type': 'candle',
            'style': s,
            'volume': volume,
            'title': 'Price Chart with Signals',
            'ylabel': 'Price',
            'ylabel_lower': 'Volume'
        }
        
        if signals:
            # Add buy signals
            buys = data[data['Signal'] == 1]
            sells = data[data['Signal'] == -1]
            
            kwargs['addplot'] = [
                mpf.make_addplot(
                    buys['Close'],
                    scatter=True,
                    markersize=100,
                    marker='^',
                    color='g'
                ),
                mpf.make_addplot(
                    sells['Close'],
                    scatter=True,
                    markersize=100,
                    marker='v',
                    color='r'
                )
            ]
        
        mpf.plot(data, **kwargs)
    
    def plot_returns_analysis(self, df: pd.DataFrame) -> None:
        """Plot returns analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Returns distribution
        sns.histplot(df['Strategy_Returns'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Returns')
        
        # QQ plot
        from scipy import stats
        stats.probplot(df['Strategy_Returns'].dropna(), dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title('Returns Q-Q Plot')
        
        # Autocorrelation
        pd.plotting.autocorrelation_plot(df['Strategy_Returns'], ax=axes[1, 0])
        axes[1, 0].set_title('Returns Autocorrelation')
        
        # Rolling volatility
        rolling_vol = df['Strategy_Returns'].rolling(window=21).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[1, 1])
        axes[1, 1].set_title('Rolling Volatility (21 days)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_analysis(self, df: pd.DataFrame) -> None:
        """Plot drawdown analysis"""
        returns = df['Strategy_Returns']
        cum_returns = (1 + returns).cumprod()
        drawdown = cum_returns / cum_returns.expanding().max() - 1
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Underwater plot
        drawdown.plot(ax=axes[0])
        axes[0].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[0].set_title('Drawdown')
        axes[0].set_ylabel('Drawdown %')
        
        # Drawdown distribution
        sns.histplot(drawdown[drawdown < 0], ax=axes[1], bins=50)
        axes[1].set_title('Drawdown Distribution')
        axes[1].set_xlabel('Drawdown %')
        
        plt.tight_layout()
        plt.show()
    
    def plot_position_analysis(self, df: pd.DataFrame) -> None:
        """Plot position analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Position sizes over time
        df['Position_Size'].plot(ax=axes[0])
        axes[0].set_title('Position Sizes')
        axes[0].set_ylabel('Position Size')
        
        # Position size distribution
        sns.histplot(df['Position_Size'][df['Position_Size'] != 0], ax=axes[1])
        axes[1].set_title('Position Size Distribution')
        axes[1].set_xlabel('Position Size')
        
        plt.tight_layout()
        plt.show()
    
    def create_tearsheet(self, 
                        df: pd.DataFrame, 
                        metrics: Dict,
                        filename: Optional[str] = None) -> None:
        """Create PDF tearsheet"""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        import io
        
        if filename is None:
            filename = 'strategy_tearsheet.pdf'
            
        # Create PDF
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont('Helvetica-Bold', 24)
        c.drawString(50, height - 50, "Strategy Analysis Tearsheet")
        
        # Add metrics
        c.setFont('Helvetica', 12)
        y = height - 100
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2%}" if abs(value) < 1 else f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            c.drawString(50, y, text)
            y -= 20
            
        # Generate and add plots
        plots = [
            (self.plot_returns_analysis, 'Returns Analysis'),
            (self.plot_drawdown_analysis, 'Drawdown Analysis'),
            (self.plot_position_analysis, 'Position Analysis')
        ]
        
        y = height - 400
        for plot_func, title in plots:
            # Create plot in memory
            plt.figure(figsize=(8, 6))
            plot_func(df)
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Add to PDF
            img = ImageReader(buf)
            c.drawImage(img, 50, y, width=400, height=300)
            y -= 350
            plt.close()
            
        c.save()

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    data = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-03")
    
    # Add sample signals and returns
    data['Signal'] = np.random.choice([-1, 0, 1], size=len(data))
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Close'].pct_change()
    
    # Initialize visualizer
    viz = Visualizer()
    
    # Create dashboard
    viz.create_strategy_dashboard(
        df=data,
        metrics={
            'Total_Return': 0.15,
            'Sharpe_Ratio': 1.5,
            'Max_Drawdown': -0.10,
            'Win_Rate': 0.55
        },
        filename='strategy_dashboard.html'
    )