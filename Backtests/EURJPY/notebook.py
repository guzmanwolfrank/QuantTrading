# Module Import

import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import oandapyV20 
import seaborn as sns 
import quantstats as qs 
import matplotlib.pyplot as plt 
%matplotlib inline 


# Account information
OANDA_ACCESS_TOKEN = "987464a00c47a9b186fcc7a93a9404a6-bf2ecb97ae681e4edeb529adef404b09"
ACCOUNT_ID = "101-001-8028197-001"
access_token = OANDA_ACCESS_TOKEN
accountID = ACCOUNT_ID
client = oandapyV20.API(access_token=access_token)
API_KEY = access_token

# OANDA API connection setup
api = oandapyV20.API(access_token=API_KEY, environment="practice")

# Function to fetch historical data
def get_historical_data(instrument, granularity, start, end):
    params = {
        "granularity": granularity,
        "from": start,
        "to": end,
    }
    request = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = api.request(request)
    data = response['candles']
    ohlc_data = [{'time': candle['time'], 'open': float(candle['mid']['o']), 'high': float(candle['mid']['h']),
                  'low': float(candle['mid']['l']), 'close': float(candle['mid']['c'])} for candle in data]
    df = pd.DataFrame(ohlc_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

# Define the instrument, granularity, and date range

granularity = 'D'  # Daily data
start_date = '01-01-2022'
end_date = '01-01-2023'

# Instrument 
#instrument = "EUR_USD"
#instrument = "EUR_AUD"
instrument = "EUR_JPY"
#instrument = "EUR_CHF"

# Granularity (daily)
granularity = "D"

# Fetch historical data
historical_data = get_historical_data(instrument, granularity, start_date, end_date)
data = historical_data
# Print the fetched data
#print(historical_data)

data.index= data.index.tz_localize(None)  # otherwise Quanstats will not be able to convert timezone or compare benchmark to strategy

#display (data)

# Add on to Data Dataframe for Profit, Units, Balance...

# Pivot calculation 
pivot= ((data['open']+data ['low'] + data['close'])/3)
data['pivot'] = pivot          
data['previous_pivot'] = data['pivot'].shift(1)

# Create a Signal column with 1, whenever the previous pivot is greater than current open 
data.loc[data['previous_pivot'] < data['open'], 'Signal'] = 1

#End Algo Note ---- use from this point above for algorithm code 



# Start Backtest 

# Once signal 1 is created at the open, buy the open of that same day.  Sell the close.    

# Calculate daily range 
data['range']= data['high']-data['low']

# Calculate daily change 
data['change']= data['close']-data['open']

# Calculate profit
data['profit'] = data.apply(lambda row: row['change'] * 100 if row['Signal'] == 1 else 0, axis=1)

profit = data ['profit']

# Initial Balance, Cumulative Profit, Balance 
initial_balance = 10000
data['cumulative_profit'] = data['profit'].cumsum()
data['balance'] = initial_balance + data['cumulative_profit']


# Calculate 'returns' column
data['returns'] = data['profit'] / data['balance']

display(data)


# Create the line chart
sns.lineplot(x="time", y="returns", data=data)

# Customize the plot
plt.title("Hypothetical Daily Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.grid(True)

# Show the plot
plt.show()

#Quanstats#  Load returns, without having to convert to a series 
returns_series = data['returns']

benchmark = qs.utils.download_returns('eurjpy=X') # note ticker format 
qs.reports.full(returns_series , benchmark)
#qs.reports.full(benchmark)
# Html Tearsheet
#qs.reports.html(returns_series , benchmark=benchmark)
