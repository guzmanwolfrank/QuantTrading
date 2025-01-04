# utils/report_generator.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import base64
from datetime import datetime
import jinja2
import io
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    """HTML Report Generator for Trading Strategies."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
        self._setup_template()
    
    def generate_report(self, 
                       df: pd.DataFrame, 
                       metrics: Dict, 
                       figures: Dict, 
                       strategy_name: str, 
                       symbol: str) -> str:
        """Generate HTML report"""
        
        if 'Strategy_Returns' not in df.columns:
            df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
        
        context = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_name': strategy_name,
            'symbol': symbol,
            'overview_metrics': self._format_overview_metrics(metrics),
            'risk_metrics': self._format_risk_metrics(metrics),
            'performance_plots': self._create_performance_plots(df),
            'risk_plots': self._create_risk_plots(df),
            'trade_plots': self._create_trade_plots(df),
            'trade_table': self._create_trade_table(df)
        }
        
        template = jinja2.Template(self.template_str)
        return template.render(**context)
    
    def save_report(self, html_content: str, filename: str = 'strategy_report.html') -> None:
        """Save HTML report to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _format_overview_metrics(self, metrics: Dict) -> List[Dict]:
        """Format overview metrics for display"""
        formatted_metrics = []
        key_metrics = [
            ('Total Return', 'Total_Return', '{:.2%}'),
            ('Sharpe Ratio', 'Sharpe_Ratio', '{:.2f}'),
            ('Win Rate', 'Win_Rate', '{:.2%}'),
            ('Profit Factor', 'Profit_Factor', '{:.2f}'),
            ('Max Drawdown', 'Max_Drawdown', '{:.2%}')
        ]
        
        for display_name, key, format_str in key_metrics:
            if key in metrics:
                formatted_metrics.append({
                    'name': display_name,
                    'value': format_str.format(metrics[key])
                })
        
        return formatted_metrics
    
    def _format_risk_metrics(self, metrics: Dict) -> List[Dict]:
        """Format risk metrics for display"""
        formatted_metrics = []
        risk_metrics = [
            ('Value at Risk (95%)', 'VaR_95', '{:.2%}'),
            ('Expected Shortfall', 'CVaR_95', '{:.2%}'),
            ('Volatility', 'Volatility', '{:.2%}'),
            ('Sortino Ratio', 'Sortino_Ratio', '{:.2f}'),
            ('Calmar Ratio', 'Calmar_Ratio', '{:.2f}')
        ]
        
        for display_name, key, format_str in risk_metrics:
            if key in metrics:
                formatted_metrics.append({
                    'name': display_name,
                    'value': format_str.format(metrics[key])
                })
        
        return formatted_metrics
    
    def _create_performance_plots(self, df: pd.DataFrame) -> str:
        """Create performance visualization plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Returns Distribution',
                'Monthly Returns',
                'Rolling Sharpe Ratio'
            ),
            vertical_spacing=0.15
        )
        
        # Equity Curve
        equity_curve = (1 + df['Strategy_Returns']).cumprod()
        fig.add_trace(
            go.Scatter(x=df.index, y=equity_curve, name='Equity'),
            row=1, col=1
        )
        
        # Returns Distribution
        fig.add_trace(
            go.Histogram(x=df['Strategy_Returns'], name='Returns', nbinsx=50),
            row=1, col=2
        )
        
        # Monthly Returns Heatmap
        monthly_returns = df['Strategy_Returns'].resample('M').sum().to_frame()
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        pivot = monthly_returns.pivot('Year', 'Month', 'Strategy_Returns')
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn'
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe Ratio
        window = 252
        rolling_returns = df['Strategy_Returns'].rolling(window=window)
        rolling_sharpe = np.sqrt(252) * (rolling_returns.mean() / rolling_returns.std())
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rolling_sharpe, name='Rolling Sharpe'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig.to_html(full_html=False, include_plotlyjs=True)
    
    def _create_risk_plots(self, df: pd.DataFrame) -> str:
        """Create risk analysis plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Drawdown',
                'Rolling Volatility',
                'Value at Risk',
                'Risk Contribution'
            ),
            vertical_spacing=0.15
        )
        
        # Drawdown
        equity_curve = (1 + df['Strategy_Returns']).cumprod()
        drawdown = equity_curve / equity_curve.expanding().max() - 1
        
        fig.add_trace(
            go.Scatter(x=df.index, y=drawdown, fill='tozeroy', name='Drawdown'),
            row=1, col=1
        )
        
        # Rolling Volatility
        rolling_vol = df['Strategy_Returns'].rolling(window=21).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rolling_vol, name='Volatility'),
            row=1, col=2
        )
        
        # Value at Risk
        fig.add_trace(
            go.Histogram(
                x=df['Strategy_Returns'],
                name='Returns',
                nbinsx=50,
                histnorm='probability'
            ),
            row=2, col=1
        )
        
        # Risk Contribution
        if 'Position_Size' in df.columns:
            risk_contrib = df['Position_Size'] * df['Strategy_Returns'].rolling(window=21).std()
            
            fig.add_trace(
                go.Scatter(x=df.index, y=risk_contrib, name='Risk Contribution'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        return fig.to_html(full_html=False, include_plotlyjs=True)
    
    def _create_trade_plots(self, df: pd.DataFrame) -> str:
        """Create trade analysis plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trade Returns Distribution',
                'Position Sizes',
                'Trade Duration Distribution',
                'Win Rate Over Time'
            ),
            vertical_spacing=0.15
        )
        
        # Trade Returns Distribution
        trade_returns = df[df['Signal'] != 0]['Strategy_Returns']
        
        fig.add_trace(
            go.Histogram(x=trade_returns, name='Trade Returns'),
            row=1, col=1
        )
        
        # Position Sizes
        if 'Position_Size' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Position_Size'],
                    name='Position Size'
                ),
                row=1, col=2
            )
        
        # Trade Duration Distribution
        if 'Signal' in df.columns:
            trade_starts = df.index[df['Signal'] != 0]
            trade_durations = []
            
            for i in range(len(trade_starts)-1):
                duration = (trade_starts[i+1] - trade_starts[i]).days
                trade_durations.append(duration)
            
            fig.add_trace(
                go.Histogram(x=trade_durations, name='Duration'),
                row=2, col=1
            )
        
        # Rolling Win Rate
        if 'Strategy_Returns' in df.columns:
            rolling_wins = (df['Strategy_Returns'] > 0).rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rolling_wins, name='Win Rate'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        return fig.to_html(full_html=False, include_plotlyjs=True)
    
    def _create_trade_table(self, df: pd.DataFrame) -> str:
        """Create HTML table of recent trades"""
        trades = df[df['Signal'] != 0].copy()
        trades['Returns'] = trades['Strategy_Returns']
        trades = trades.tail(10)  # Last 10 trades
        
        table_html = """
        <table>
            <tr>
                <th>Date</th>
                <th>Direction</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>Return</th>
            </tr>
        """
        
        for idx, trade in trades.iterrows():
            direction = 'Long' if trade['Signal'] == 1 else 'Short'
            return_class = 'positive' if trade['Returns'] > 0 else 'negative'
            
            table_html += f"""
            <tr>
                <td>{idx.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{direction}</td>
                <td>{trade['Close']:.4f}</td>
                <td>{trade['Close']:.4f}</td>
                <td class="{return_class}">{trade['Returns']:.2%}</td>
            </tr>
            """
            
        table_html += "</table>"
        return table_html
    
    def _setup_template(self) -> None:
        """Setup HTML template"""
        self.template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-title {
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 5px;
                }
                .metric-value {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #333;
                }
                .plot-container {
                    margin-bottom: 30px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                .positive {
                    color: #2ecc71;
                }
                .negative {
                    color: #e74c3c;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trading Strategy Report</h1>
                    <p>Generated on: {{ generation_time }}</p>
                    <p>Strategy: {{ strategy_name }} | Symbol: {{ symbol }}</p>
                </div>
                
                <div class="section">
                    <h2>Strategy Overview</h2>
                    <div class="metrics-grid">
                        {% for metric in overview_metrics %}
                        <div class="metric-card">
                            <div class="metric-title">{{ metric.name }}</div>
                            <div class="metric-value {% if metric.value < 0 %}negative{% endif %}
                                                   {% if metric.value > 0 %}positive{% endif %}">
                                {{ metric.value }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Charts</h2>
                    {{ performance_plots | safe }}
                </div>
                
                <div class="section">
                    <h2>Risk Analysis</h2>
                    {{ risk_plots | safe }}
                    <div class="metrics-grid">
                        {% for metric in risk_metrics %}
                        <div class="metric-card">
                            <div class="metric-title">{{ metric.name }}</div>
                            <div class="metric-value">{{ metric.value }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                <div class="section">
                    <h2>Trade Analysis</h2>
                    {{ trade_plots | safe }}
                    <h3>Recent Trades</h3>
                    {{ trade_table | safe }}
                </div>
            </div>
        </body>
        </html>
        """
    
    def generate_pdf_report(self,
                          df: pd.DataFrame,
                          metrics: Dict,
                          filename: str = 'strategy_report.pdf') -> None:
        """
        Generate PDF report
        
        Parameters:
        -----------
        df : pd.DataFrame
            Trading data with signals and returns
        metrics : Dict
            Performance metrics
        filename : str
            Output filename for PDF
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        elements.append(Paragraph("Trading Strategy Report", styles['Heading1']))
        
        # Add performance metrics table
        data = [["Metric", "Value"]]
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 1:  # Percentage format
                    formatted_value = f"{value:.2%}"
                else:  # Regular float format
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            data.append([key, formatted_value])
        
        # Create table with styling
        table = Table(data, colWidths=[4*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    data = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-03")
    
    # Add sample signals and returns
    data['Signal'] = np.random.choice([-1, 0, 1], size=len(data))
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Close'].pct_change()
    
    # Sample metrics
    metrics = {
        'Total_Return': 0.15,
        'Sharpe_Ratio': 1.5,
        'Max_Drawdown': -0.10,
        'Win_Rate': 0.55,
        'Volatility': 0.12,
        'Sortino_Ratio': 2.1
    }
    
    # Initialize report generator
    report_gen = ReportGenerator()
    
    # Generate reports
    html_report = report_gen.generate_report(
        df=data,
        metrics=metrics,
        figures={},
        strategy_name="Sample Strategy",
        symbol="EURUSD"
    )
    
    # Save reports
    report_gen.save_report(html_report, 'strategy_report.html')
    report_gen.generate_pdf_report(data, metrics, 'strategy_report.pdf')