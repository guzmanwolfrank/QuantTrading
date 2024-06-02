import yfinance as yf
import quantstats as qs
import warnings
import pandas as pd 
import os
# Ignore warnings
warnings.filterwarnings("ignore")

# Download USD/JPY data
data = yf.download("USDJPY=X", start="2020-01-01", end="2021-01-01")

# Calculate moving averages
data['50_day_MA'] = data['Open'].rolling(window=50).mean()

# Generate sell signals
data['Signal'] = 0
data.loc[data['Open'] < data['50_day_MA'], 'Signal'] = -1  # Sell signal

# Backtest the strategy
data['Pnl'] = data['Signal'].shift(1) * (data['Open'].shift(-1) - data['Open'])

# Calculate cumulative returns and other performance metrics
qs.extend_pandas()
returns = data['Pnl']
#qs.reports.html(returns)
qs.reports.metrics(returns)
display(data)

csv_file_path = 'data/backtest_results.csv'  # Specify the file path
data.to_csv(csv_file_path)

# Print the file path where the CSV was saved
print(f"CSV saved to: {os.path.abspath(csv_file_path)}")

