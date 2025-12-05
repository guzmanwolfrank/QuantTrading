# ATR Analysis for Futures Contracts
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import HTML, display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import webbrowser

# Define the periods for ATR calculation
PERIODS = [3, 5, 10, 20, 50, 100, 200]
TICKERS = ["MNQ=F", "NQ=F"]

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)"""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    return atr

def fetch_data(ticker_symbol, lookback_days=250, max_retries=3):
    """Fetch historical data with retry mechanism"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Fetching data for {ticker_symbol}, attempt {attempt}...")
            ticker = yf.Ticker(ticker_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise Exception(f"Empty dataframe returned for {ticker_symbol}")
                
            print(f"Successfully fetched data for {ticker_symbol}")
            return df
            
        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print(f"Failed to fetch data for {ticker_symbol} after {max_retries} attempts")
    return None

def analyze_ticker(ticker_symbol):
    """Analyze ATR for a given ticker"""
    print(f"\n--- Analyzing {ticker_symbol} ---")
    
    # Fetch data
    df = fetch_data(ticker_symbol)
    if df is None or df.empty:
        print(f"No data available for {ticker_symbol}")
        return None
    
    # Calculate ATR for different periods
    results = {
        'ticker': ticker_symbol,
        'latest_close': df['Close'].iloc[-1],
        'price_time': df.index[-1],  # Store the timestamp of the latest price
        'atr_values': {},
        'atr_percentage': {},
        'data': df
    }
    
    # Calculate ATR for each period
    for period in PERIODS:
        df[f'ATR_{period}'] = calculate_atr(df['High'], df['Low'], df['Close'], period)
        
        # Store the latest values
        latest_atr = df[f'ATR_{period}'].iloc[-1]
        results['atr_values'][period] = latest_atr
        results['atr_percentage'][period] = (latest_atr / results['latest_close']) * 100
    
    return results

def create_comparison_chart(results):
    """Create a comparison chart for ATR periods"""
    ticker = results['ticker']
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2, 
                      subplot_titles=(f"ATR Values - {ticker}", f"ATR % of Price - {ticker}"))
    
    # Prepare data
    periods = list(results['atr_values'].keys())
    atr_values = list(results['atr_values'].values())
    atr_percentages = list(results['atr_percentage'].values())
    
    # Color gradient from blue to green
    colors = ['rgba(0, 150, 255, 0.8)', 'rgba(0, 170, 255, 0.8)', 
              'rgba(0, 190, 255, 0.8)', 'rgba(0, 210, 255, 0.8)', 
              'rgba(0, 230, 255, 0.8)', 'rgba(0, 250, 255, 0.8)', 
              'rgba(0, 255, 240, 0.8)']
    
    # Add bars for ATR values
    fig.add_trace(
        go.Bar(
            x=[str(p) for p in periods],
            y=atr_values,
            name="ATR Value",
            marker_color=colors,
            hovertemplate="Period: %{x} days<br>ATR: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add bars for ATR percentages
    fig.add_trace(
        go.Bar(
            x=[str(p) for p in periods],
            y=atr_percentages,
            name="ATR %",
            marker_color=colors,
            hovertemplate="Period: %{x} days<br>ATR: %{y:.2f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#1E1E1E",
        font=dict(family="Arial", size=12, color="white"),
        height=400,
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Period (Days)", row=1, col=1)
    fig.update_xaxes(title_text="Period (Days)", row=1, col=2)
    fig.update_yaxes(title_text="ATR Value", row=1, col=1)
    fig.update_yaxes(title_text="ATR % of Price", row=1, col=2)
    
    return fig

def create_historical_chart(results):
    """Create historical ATR chart"""
    ticker = results['ticker']
    df = results['data'].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add price line (as area)
    fig.add_trace(
        go.Scatter(
            x=df.index[-90:],  # Last 90 days
            y=df['Close'][-90:],
            name="Price",
            line=dict(color='rgba(0, 255, 255, 0.8)', width=1),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 255, 0.1)'
        )
    )
    
    # Add ATR lines for each period
    colors = ['#7EB6FF', '#5A9CFF', '#3D83FF', '#0066FF', '#0052CC', '#003D99', '#002966']
    
    for i, period in enumerate(PERIODS):
        fig.add_trace(
            go.Scatter(
                x=df.index[-90:],  # Last 90 days
                y=df[f'ATR_{period}'][-90:],
                name=f"ATR-{period}",
                line=dict(color=colors[i], width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#1E1E1E",
        font=dict(family="Arial", size=12, color="white"),
        title=f"{ticker} Price and ATR (Last 90 Days)",
        height=500,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_html_report(analysis_results):
    """Generate an HTML report with the analysis results"""
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #121212;
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #333;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #00FFFF;
            margin-bottom: 10px;
        }
        .header p {
            color: #999;
            font-size: 1.1em;
        }
        .credits {
            font-size: 1em;
            color: #00FFFF;
            margin-top: 5px;
        }
        .section {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .section-header h2 {
            color: #00FFFF;
            margin: 0;
        }
        .price-info {
            font-size: 1.2em;
            padding: 10px;
            border-radius: 5px;
            background-color: #2A2A2A;
        }
        .summary {
            margin: 20px 0;
            padding: 15px;
            background-color: #2A2A2A;
            border-radius: 5px;
            line-height: 1.6;
        }
        .atr-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border-radius: 5px;
            overflow: hidden;
        }
        .atr-table th, .atr-table td {
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        .atr-table th {
            background-color: #333;
            color: #00FFFF;
        }
        .atr-table tr {
            background-color: #2A2A2A;
        }
        .atr-table tr:hover {
            background-color: #383838;
        }
        .period-label {
            text-align: left;
            font-weight: bold;
        }
        .low-volatility {
            color: #4CAF50;
        }
        .medium-volatility {
            color: #FFC107;
        }
        .high-volatility {
            color: #F44336;
        }
        .chart-container {
            margin: 30px 0;
            height: 400px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
    """
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Futures ATR Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Futures ATR Analysis</h1>
                <p>Average True Range analysis for MNQ and NQ futures contracts</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="credits">Made by @ghostokamiiii @guzmanwolfrank:github</p>
            </div>
    """
    
    # Process each ticker result
    for ticker_symbol, result in analysis_results.items():
        if result is None:
            html += f"""
            <div class="section">
                <div class="section-header">
                    <h2>{ticker_symbol}</h2>
                </div>
                <div class="summary">
                    <p>No data available for this ticker. Please check the ticker symbol or try again later.</p>
                </div>
            </div>
            """
            continue
        
        # For valid results, create detailed section
        ref_period = 10  # Using 10-day ATR as reference
        
        html += f"""
        <div class="section">
            <div class="section-header">
                <h2>{result['ticker']}</h2>
                <div class="price-info">Current Price as of {result['price_time'].strftime('%Y-%m-%d %H:%M:%S')}: ${result['latest_close']:.2f}</div>
            </div>
            
            <div class="summary">
                <strong>Volatility Analysis:</strong> The {result['ticker']} futures contract is currently showing 
                <span class="{
                'low-volatility' if result['atr_percentage'][ref_period] < 0.5 else 
                'medium-volatility' if result['atr_percentage'][ref_period] < 1.0 else 
                'high-volatility'
                }">
                {
                'low' if result['atr_percentage'][ref_period] < 0.5 else 
                'moderate' if result['atr_percentage'][ref_period] < 1.0 else 
                'high'
                } volatility</span> with a {ref_period}-day ATR of {result['atr_values'][ref_period]:.2f} points 
                ({result['atr_percentage'][ref_period]:.2f}% of current price).
                
                The short-term volatility (3-day ATR) is 
                {
                'lower' if result['atr_values'][3] < result['atr_values'][ref_period] else 
                'similar to' if abs(result['atr_values'][3] - result['atr_values'][ref_period]) / result['atr_values'][ref_period] < 0.1 else 
                'higher'
                } 
                than the medium-term volatility, indicating 
                {
                'decreasing' if result['atr_values'][3] < result['atr_values'][ref_period] else 
                'stable' if abs(result['atr_values'][3] - result['atr_values'][ref_period]) / result['atr_values'][ref_period] < 0.1 else 
                'increasing'
                }
                price movement.
            </div>
            
            <table class="atr-table">
                <tr>
                    <th>Period (Days)</th>
                    <th>ATR Value</th>
                    <th>ATR % of Price</th>
                    <th>Volatility Level</th>
                </tr>
        """
        
        # Add table rows for each ATR period
        for period in PERIODS:
            atr_value = result['atr_values'][period]
            atr_pct = result['atr_percentage'][period]
            
            volatility_class = 'low-volatility' if atr_pct < 0.5 else 'medium-volatility' if atr_pct < 1.0 else 'high-volatility'
            volatility_text = 'Low' if atr_pct < 0.5 else 'Medium' if atr_pct < 1.0 else 'High'
            
            html += f"""
                <tr>
                    <td class="period-label">{period}</td>
                    <td>{atr_value:.2f}</td>
                    <td>{atr_pct:.2f}%</td>
                    <td class="{volatility_class}">{volatility_text}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <div class="chart-container" id="comparison-{result['ticker']}">
                <!-- Comparison chart placeholder -->
            </div>
            
            <div class="chart-container" id="historical-{result['ticker']}">
                <!-- Historical chart placeholder -->
            </div>
        </div>
        """
    
    # Add footer
    html += """
            <div class="footer">
                <p>ATR (Average True Range) is a technical indicator that measures market volatility by decomposing the entire range of an asset price for that period.</p>
                <p>Higher ATR values indicate higher volatility, which may suggest potential for larger price movements.</p>
            </div>
        </div>
        
        <script>
            // JavaScript for charts will be added dynamically
        </script>
    </body>
    </html>
    """
    
    return html

def main():
    """Main function to run the analysis"""
    print("Starting ATR analysis for futures contracts...")
    
    # Analyze tickers
    analysis_results = {}
    for ticker in TICKERS:
        analysis_results[ticker] = analyze_ticker(ticker)
    
    # Generate charts and HTML
    charts_js = "<script>"
    
    for ticker, result in analysis_results.items():
        if result is not None:
            # Create comparison chart
            comparison_chart = create_comparison_chart(result)
            charts_js += f"""
                var comparisonData_{ticker.replace('=', '_')} = {comparison_chart.to_json()};
                Plotly.newPlot('comparison-{ticker}', 
                               comparisonData_{ticker.replace('=', '_')}.data, 
                               comparisonData_{ticker.replace('=', '_')}.layout);
            """
            
            # Create historical chart
            historical_chart = create_historical_chart(result)
            charts_js += f"""
                var historicalData_{ticker.replace('=', '_')} = {historical_chart.to_json()};
                Plotly.newPlot('historical-{ticker}', 
                               historicalData_{ticker.replace('=', '_')}.data, 
                               historicalData_{ticker.replace('=', '_')}.layout);
            """
    
    charts_js += "</script>"
    
    # Generate final HTML report
    html_report = generate_html_report(analysis_results)
    html_report = html_report.replace('<script>\n            // JavaScript for charts will be added dynamically\n        </script>', charts_js)
    
    # Save the HTML report to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"futures_atr_analysis_{timestamp}.html"
    file_path = os.path.abspath(filename)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print(f"\nHTML report saved to: {filename}")
    print(f"Full path: {file_path}")
    
    # Open the HTML file in the default web browser
    print("\nOpening report in default web browser...")
    webbrowser.open('file://' + file_path)
    
    # Also display in the notebook
    display(HTML(html_report))
    
    print("Analysis complete!")
    return html_report

# Run the analysis when executing the notebook
if __name__ == "__main__":
    main()