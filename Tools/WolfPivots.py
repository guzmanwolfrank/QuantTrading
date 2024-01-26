import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



# Tickers 
# NQ 100 Emini Futures = 'NQ=F'
# SP 500 Emini Futures = 'ES=F'
# YM Dow Emini Futures = 'YM=F'
#  EURUSD FOREX = 'EURUSD = X' 
# GOLD ETF = 'GLD'
# SP 500 ETF = 'SPY' 





# Get user input for the stock ticker
ticker = 'NQ=F'



# Download data for the last 30 days with daily intervals
ndata = yf.download(ticker, period="30d", interval="1d")

# Display the downloaded data
#display(ndata)

# Extract high, low, close, and open prices and round them to 2 decimal places
high = ndata['High'].round(2)
low = ndata['Low'].round(2)
close = ndata['Close'].round(2)
open_price = ndata['Open'].round(2)

# Calculate the pivot point and support/resistance levels
pivot_point = (high + low + close) / 3
support1 = (2 * pivot_point) - high
support2 = pivot_point - (high - low)
support3 = low - 2 * (high - pivot_point)
resistance1 = (2 * pivot_point) - low
resistance2 = pivot_point + (high - low)
resistance3 = high + 2 * (pivot_point - low)

# Create a DataFrame with the calculated values
pivot_data = pd.DataFrame({
    'High': high,
    'Low': low,
    'Close': close,
    'Open': open_price,
  
    'R3': resistance3,
    'R2': resistance2,
    'R1': resistance1,

    'Pivot_Point': pivot_point,
  
    'S1': support1,
    'S2': support2,
    'S3': support3,
   
}).round(2)

# Display the DataFrame with calculated values
# Display(pivot_data)
df_except_last_row = pivot_data.iloc[:-1]
Pivdata= df_except_last_row


dpiv = (Pivdata.iloc[-1])

print(ticker, dpiv)


# Download data for the last 30 days with daily intervals
ndata = yf.download(ticker, period="30d", interval="1d")

# Extract high, low, close, and open prices and round them to 2 decimal places
high = ndata['High'].round(2)
low = ndata['Low'].round(2)
close = ndata['Close'].round(2)
open = ndata['Open'].round(2)
volume = ndata['Volume'] 

# Calculate the pivot point and support/resistance levels
pivot_point = (high + low + close) / 3
support1 = (2 * pivot_point) - high
support2 = pivot_point - (high - low)
support3 = low - 2 * (high - pivot_point)
resistance1 = (2 * pivot_point) - low
resistance2 = pivot_point + (high - low)
resistance3 = high + 2 * (pivot_point - low)

# Create a DataFrame with the calculated values
pivot_data = pd.DataFrame({
    'High': high,
    'Low': low,
    'Close': close,
    'Open': open_price,
    'R3': resistance3,
    'R2': resistance2,
    'R1': resistance1,
    'Pivot_Point': pivot_point,
    'S1': support1,
    'S2': support2,
    'S3': support3,
    'Volume': volume,
    
}).round(2)

# Exclude the last row for plotting
df_except_last_row = pivot_data.iloc[:-1]

print(Pivdata)

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_except_last_row[['Close', 'R3', 'R2', 'R1', 'Pivot_Point', 'S1', 'S2', 'S3']])
plt.title('NQ 100 CME Emini Futures Price and Pivot Points')
plt.xlabel('Date')
plt.ylabel('Price')

# Rotate y-axis labels by 90 degrees
plt.xticks(rotation=90)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


