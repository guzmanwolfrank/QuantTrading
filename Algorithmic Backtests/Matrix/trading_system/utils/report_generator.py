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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

class ReportGenerator:
    """HTML Report Generator for Trading Strategies."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
        self._setup_template()
    
    def _setup_template(self):
        """Setup Jinja2 template"""
        if self.template_path:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                self.template_str = f.read()
        else:
            # Default minimal HTML template if no path provided
            self.template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Strategy Report</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .positive { color: green; }
                    .negative { color: red; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>{{ strategy_name }} Strategy Report</h1>
                <p>Generated on: {{ generation_time }}</p>
                
                <h2>Overview Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for metric in overview_metrics %}
                    <tr>
                        <td>{{ metric.name }}</td>
                        <td>{{ metric.value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h2>Risk Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for metric in risk_metrics %}
                    <tr>
                        <td>{{ metric.name }}</td>
                        <td>{{ metric.value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h2>Performance Plots</h2>
                {{ performance_plots|safe }}
                
                <h2>Risk Plots</h2>
                {{ risk_plots|safe }}
                
                <h2>Trade Plots</h2>
                {{ trade_plots|safe }}
                
                <h2>Recent Trades</h2>
                {{ trade_table|safe }}
            </body>
            </html>
            """
    
def generate_report(self, 
                   df: pd.DataFrame, 
                   metrics: Dict, 
                   figures: Dict, 
                   strategy_name: str, 
                   symbol: str) -> str:
    """Generate HTML report with all analysis components"""
    
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
    
    # DEBUG: Print out context details
    print("Context Keys:", list(context.keys()))
    print("\nOverview Metrics:")
    for metric in context['overview_metrics']:
        print(f"{metric['name']}: value={metric['value']}, raw_value={metric['raw_value']}, type={type(metric['raw_value'])}")
    
    print("\nRisk Metrics:")
    for metric in context['risk_metrics']:
        print(f"{metric['name']}: value={metric['value']}, raw_value={metric['raw_value']}, type={type(metric['raw_value'])}")
    
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
            try:
                value = metrics.get(key, 0)
                formatted_metrics.append({
                    'name': display_name,
                    'value': format_str.format(value),
                    'raw_value': float(value)  # Ensure numeric value for comparison
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing overview metric {key}: {e}")
        
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
            try:
                value = metrics.get(key, 0)
                formatted_metrics.append({
                    'name': display_name,
                    'value': format_str.format(value),
                    'raw_value': float(value)  # Ensure numeric value for comparison
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing risk metric {key}: {e}")
        
        return formatted_metrics
    
    # ... [rest of the file remains the same as in the previous implementation] ...

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
        'Sortino_Ratio': 2.1,
        'VaR_95': 0.05,
        'CVaR_95': 0.07,
        'Calmar_Ratio': 1.8
    }
    
    # Initialize report generator
    report_gen = ReportGenerator()
    
    # Generate and save reports
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