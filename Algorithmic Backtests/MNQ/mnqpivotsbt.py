import yfinance as yf
import quantstats as qs
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Set the ticker symbol and time period
ticker_symbol = "MNQ=F"
start_date = "2024-03-01"
end_date = "2024-03-11"

# Fetch intraday data using yfinance for the last 10 days with 15-minute intervals
data = yf.download(ticker_symbol, period="30d", interval="15m").round(2)

# Calculate Pivot Points for 15-minute intervals
data['Pivot_Point'] = (data['High'] + data['Low'] + data['Close']) / 3
data['Support_1'] = 2 * data['Pivot_Point'] - data['High']
data['Resistance_1'] = 2 * data['Pivot_Point'] - data['Low']
data['Support_2'] = data['Pivot_Point'] - (data['High'] - data['Low'])
data['Resistance_2'] = data['Pivot_Point'] + (data['High'] - data['Low'])
data['Support_3'] = data['Low'] - 2 * (data['High'] - data['Pivot_Point'])
data['Resistance_3'] = data['High'] + 2 * (data['Pivot_Point'] - data['Low'])

# Calculate 10 period moving average for the close price
data['10_Day_MA'] = data['Close'].rolling(window=10).mean()

# Round numbers
data = data.round(2)

# Create a Signal column with 1, whenever the pivot is greater than current open// 
data.loc[data['Pivot_Point'] > data['Open'], 'Signal'] = 1

# Calculate daily range
data['Range'] = data['High'] - data['Low']

# Calculate daily change
data['Change'] = data['Close'] - data['Open']

# Calculate profit // MNQ is 2 dollars per point // Position is opened on Open, and closed on same day close, this is why we use CHANGE 
# for calculating the profit. 
data['Profit'] = data.apply(lambda row: row['Change'] * 2 if row['Signal'] == 1 else 0, axis=1)

profit = data['Profit']

# Initial Balance, Cumulative Profit, Balance
initial_balance = 10000
data['cumulative_profit'] = data['Profit'].cumsum()
data['Balance'] = initial_balance + data['cumulative_profit']

# Calculate 'returns' column
data['returns'] = data['Profit'] / data['Balance']

# Display the data
print(data)

# Display performance metrics using quantstats
qs.reports.full(data['returns'])
#Quanstats#  Load returns, without having to convert to a series 
returns_series = data['returns']

benchmark = qs.utils.download_returns('MNQ=F') # note ticker format 
qs.reports.full(returns_series , benchmark)
#qs.reports.full(benchmark)
# Html Tearsheet
qs.reports.html(returns_series , benchmark=benchmark)