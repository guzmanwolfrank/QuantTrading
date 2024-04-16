import yfinance as yf
import pandas as pd
import quantstats as qs
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Download historical data for NQ=F
ticker = 'NQ=F'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['UpperBB'] = data['SMA'] + (data['STD'] * num_std)
    data['LowerBB'] = data['SMA'] - (data['STD'] * num_std)
    return data

data = calculate_bollinger_bands(data)

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

data = calculate_rsi(data)

# Backtest Strategy
def backtest_strategy(data):
    positions = []
    for i in range(len(data)):
        if data['Open'][i] >= data['LowerBB'][i] and data['RSI'][i] <= 70:
            positions.append('Sell')
        elif data['Open'][i] <= data['UpperBB'][i] and data['RSI'][i] >= 30:
            positions.append('Buy')
        else:
            positions.append('')
    data['Position'] = positions
    return data

data = backtest_strategy(data)

# Calculate Returns
#data['Returns'] = data['Close'].pct_change() if buy_signals, sell signals 

# Filter Trades
buy_signals = data[data['Position'] == 'Buy']
sell_signals = data[data['Position'] == 'Sell']
df = data 
# Show head of Data Dataframe 
#print(df.head(5))
#print(buy_signals.head(5))
#print(sell_signals.head(5))

df['Range']= df['High']-df['Low']

# What is the current average daily range from high to low on NQ Futures since Jan 2020? 
average = df['Range'].mean()
# Print the average value
print("Average of the Range:", average)
# Now we need to calculate Balance, Profit, and Returns columns for this dataframe and we are all set. 

# Points, Lots, Profit, Balance, Returns 
# Calculate Points
def calculate_points(data):
    points = []
    for i in range(len(data)):
        if data['Position'][i] == 'Buy':
            points.append(data['Close'][i] - data['Open'][i])
        elif data['Position'][i] == 'Sell':
            points.append(data['Open'][i] - data['Close'][i])
        else:
            points.append(0)
    data['Points'] = points
    return data

df = calculate_points(df)

# Define Lot Size
df['Lots']= 1

# Calculate commissions column

df['commissions'] = 1.98 * df['Lots']


# Define Profits 
df['Profit']= df['Points']*df['Lots']-df['commissions'].round(2)

# Initial Balance, Cumulative Profit, Balance

df['cumulative_profit'] = df['Profit'].cumsum()
data['Balance'] = 10000 + df['cumulative_profit'].round(2)

# Calculate 'returns' column
df['returns'] = df['Profit'] / df['Balance']

print(df)
#Quanstats#  Load returns, without having to convert to a series 
returns_series = df['returns']

benchmark = qs.utils.download_returns('NQ=F') # note ticker format 
qs.reports.full(returns_series , benchmark)
# Html Tearsheet3
#qs.reports.html(returns_series , benchmark=benchmark)
# Calculate total profit
total_profit = df['Profit'].sum()
print("Total Profit:", total_profit)

# Calculate total returns
total_returns = df['Profit'].sum() / df['Balance'].iloc[-1]
print("Total Returns:", total_returns)

# Calculate annualized returns
years = (df.index[-1] - df.index[0]).days / 365.25
annualized_returns = (1 + total_returns) ** (1 / years) - 1
print("Annualized Returns:", annualized_returns)

# Calculate Sharpe Ratio
sharpe_ratio = qs.stats.sharpe(returns_series)
print("Sharpe Ratio:", sharpe_ratio)

# Calculate Sortino Ratio
sortino_ratio = qs.stats.sortino(returns_series)
print("Sortino Ratio:", sortino_ratio)

# Maximum Drawdown
max_drawdown = qs.stats.max_drawdown(returns_series)
print("Maximum Drawdown:", max_drawdown)

# Plotting
qs.plots.snapshot(returns_series)

# Generate a full report including tearsheet
#qs.reports.full(returns_series, benchmark)

# If you want to save the report to HTML:
# qs.reports.html(returns_series, benchmark=benchmark, output='report.html')

import os

# Get the path to your desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# Save the DataFrame as a CSV file on your desktop
#df.to_csv(os.path.join(desktop_path, 'bollidf.csv'), index=False)

#print("DataFrame 'df' has been saved as 'df.csv' on your desktop.")