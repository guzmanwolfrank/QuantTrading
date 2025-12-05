# Futures ATR Analysis Tool

## Overview
This tool provides a comprehensive analysis of Average True Range (ATR) metrics for MNQ and NQ futures contracts across multiple time periods. It generates an interactive HTML dashboard that visualizes volatility patterns to help traders make more informed decisions.

## Created By
- [@ghostokamiiii](https://twitter.com/ghostokamiiii)
- [@guzmanwolfrank:github](https://github.com/guzmanwolfrank)

## Features
- **Multi-timeframe Analysis**: Calculates ATR for 3, 5, 10, 20, 50, 100, and 200-day periods
- **Interactive Visualizations**: Dynamic Plotly charts for comparing ATR values
- **Historical Context**: Displays 90-day historical ATR trends
- **Volatility Categorization**: Color-coded indicators for low, medium, and high volatility
- **Dark-themed UI**: Optimized for extended viewing with black background and white text
- **Automatic Browser Display**: Opens the report in your default web browser

![_C__Users_Wolfrank_Desktop_futures_atr_analysis_20250327_155644 html(iPhone 12 Pro)](https://github.com/user-attachments/assets/502d5d28-524c-4be1-98e0-ee192bbf83ad)

## Requirements
The script requires the following Python packages:
```
yfinance
pandas
numpy
matplotlib
plotly
IPython
```

You can install them using pip:
```bash
pip install yfinance pandas numpy matplotlib plotly
```

## Usage

### Option 1: Running in Jupyter Notebook
1. Copy the entire script into a Jupyter notebook cell
2. Run the cell
3. The analysis will automatically:
   - Fetch the latest data for MNQ and NQ futures
   - Calculate ATR metrics
   - Generate an HTML report
   - Save the file with a timestamp
   - Open the report in your default browser

### Option 2: Running as a Python Script
1. Save the code as `futures_atr_analysis.py`
2. Run from the command line:
   ```bash
   python futures_atr_analysis.py
   ```
3. View the generated HTML report in your browser

## Understanding the Output

### ATR Values
- **Higher ATR**: Indicates increased volatility or price movement
- **Lower ATR**: Suggests more stable or consolidated price action

### Volatility Categories
- **Low Volatility** (Green): ATR less than 0.5% of price
- **Medium Volatility** (Yellow): ATR between 0.5% and 1.0% of price
- **High Volatility** (Red): ATR greater than 1.0% of price

### Comparison Insights
- Short-term vs. Medium-term volatility comparison helps identify volatility trends
- Historical chart shows how volatility has evolved over the past 90 days

## Interpreting the Results
- **Increasing ATR**: May signal the start of a new trend or breakout
- **Decreasing ATR**: Often indicates consolidation or diminishing momentum
- **ATR Spikes**: Could indicate significant news events or market shifts
- **Low ATR Following High ATR**: Often signals a potential reversal or continuation pattern

## Customization
You can modify the script to:
- Analyze different futures contracts by changing the `TICKERS` list
- Adjust the lookback periods by modifying the `PERIODS` list
- Change the color scheme in the various chart creation functions
- Add additional technical indicators to enhance the analysis

## Troubleshooting
- If Yahoo Finance data retrieval fails, the script implements retry logic with exponential backoff
- Ensure your internet connection is stable
- If charts don't display properly, check that you have Plotly installed correctly

## License
This project is available for open use with proper attribution to the creators.

---

*Note: This tool is for informational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions.*
