import yfinance as yf
import pandas as pd
import warnings 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime






# Ignore all warnings
warnings.filterwarnings('ignore')


# Set the ticker symbol 
ticker_symbol = "MNQ=F"
# Fetch intraday data using yfinance for the last 60 days with 15-minute intervals
data = yf.download(ticker_symbol, period="60d", interval="15m").round(2)



# Round numbers
data = data.round(2)
#display(data)
# Convert the index to datetime format
data.index = pd.to_datetime(data.index)

# Filter rows for the specific times 09:30 and 10:15
filtered_data = data[(data.index.time == pd.to_datetime("09:30:00").time()) | (data.index.time == pd.to_datetime("10:15:00").time())]
#display(filtered_data)

# Group by date and aggregate the data
combined_data = filtered_data.groupby(filtered_data.index.date).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# Calculate the range and add it as a new column
combined_data['Range'] = combined_data['High'] - combined_data['Low']

# Round numbers
data = data.round(2)
cdata = combined_data.round(2)
#display(cdata)

# Calculate the average range for different windows of days
windows = [5, 10, 20, 30, 40, 50, 60]
for window in windows:
    avg_range = combined_data['Range'].rolling(window=window).mean().iloc[-1].round(2)
   # print(f"Average Opening Range for last {window} days:", avg_range)

# Calculate the average of the 'Range' column
average_range = combined_data['Range'].mean()

Ddata = yf.download(ticker_symbol, period="60d", interval="1d").round(2)
Ddata['ABSDayChange']= abs(Ddata['Close']- Ddata['Open'])
Ddata['DayChange']= (Ddata['Close']- Ddata['Open'])
Ddata['OpenOnlyRange']= combined_data['Range']
Ddata['DayRange']= Ddata['High']-Ddata['Low']

# Calculate absolute day change
Ddata['ABSDayChange'] = abs(Ddata['Close'] - Ddata['Open'])

# Calculate day change
Ddata['DayChange'] = Ddata['Close'] - Ddata['Open']

# Calculate OpenOnlyRange
Ddata['OpenOnlyRange'] = combined_data['Range']

# Calculate DayRange
Ddata['DayRange'] = Ddata['High'] - Ddata['Low']

# Define the window sizes
windows = [5, 10, 20, 30, 40, 50]


print("Average Open Range:", average_range)


# Calculate rolling averages for each column and window size
for column in ['ABSDayChange', 'OpenOnlyRange', 'DayRange']:
    print(f"Rolling Averages for {column}:")
    for window in windows:
        rolling_avg = Ddata[column].rolling(window=window).mean().iloc[-1]
        print(f" - {window}-day Average: {rolling_avg:.2f}")
    print()
    

display(Ddata)

# Define the window sizes
windows = [5, 10, 20, 30, 40, 50]

# Calculate rolling averages for each column and window size
rolling_avgs = {}
for column in ['ABSDayChange', 'OpenOnlyRange', 'DayRange']:
    rolling_avgs[column] = {}
    for window in windows:
        rolling_avg = Ddata[column].rolling(window=window).mean().iloc[-1]
        rolling_avgs[column][window] = rolling_avg

# Create a DataFrame for rolling averages
rolling_avg_df = pd.DataFrame(rolling_avgs)

# Plot line chart comparing the rolling averages
plt.figure(figsize=(10, 6))
sns.lineplot(data=rolling_avg_df, dashes=False)
plt.title('Rolling Averages Comparison')
plt.xlabel('Window Size (days)')
plt.ylabel('Value')
plt.legend(title='Metric')
plt.xticks(windows)
plt.show()


display(rolling_avg_df)
#New Questions:  What percentage of the day range is the open only range?  

#To answer this question we must make a new column, OpenOnly_as_PCT_of_DayRange
Ddata['OpenOnly_as_PCT_of_DayRange']= Ddata['OpenOnlyRange']/Ddata['DayRange']
PCT_of = Ddata['OpenOnly_as_PCT_of_DayRange']

display(Ddata)

# Count instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 35%
count_greater_than_35 = (Ddata['OpenOnly_as_PCT_of_DayRange'] > 0.35).sum()

# Count instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 50%
count_greater_than_50 = (Ddata['OpenOnly_as_PCT_of_DayRange'] > 0.50).sum()

# Count instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 70%
count_greater_than_70 = (Ddata['OpenOnly_as_PCT_of_DayRange'] > 0.70).sum()

print("Number of instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 35%:", count_greater_than_35)
print("Number of instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 50%:", count_greater_than_50)
print("Number of instances where 'OpenOnly_as_PCT_of_DayRange' is greater than 70%:", count_greater_than_70)


# Calculate the total number of instances in the DataFrame Ddata
total_instances = len(Ddata)

print("Total number of instances in the DataFrame:", total_instances)

# Download historical data for MNQ
ticker = "MNQ=F"  # Ticker symbol for Micro E-mini NASDAQ-100 Futures
start_date = datetime.datetime.now() - datetime.timedelta(days=365)  # One year of historical data
end_date = datetime.datetime.now()

data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess data
data = data.dropna()  # Remove any rows with missing values
data['Date'] = data.index  # Convert index to a column
data['Date'] = data['Date'].apply(lambda x: x.toordinal())  # Convert dates to ordinal values

# Define features (predictors) and target (closing prices)
X = data[['Date']].values
y = data['Close'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for the next ten days
next_ten_days = np.arange(max(X), max(X) + 11).reshape(-1, 1)
predicted_prices = model.predict(next_ten_days).round(2)

# Convert ordinal dates back to datetime objects
next_ten_dates = [datetime.date.fromordinal(int(date)) for date in next_ten_days.flatten()]

# Print the predicted closing prices with corresponding dates for the next ten days
print("Predicted closing prices for the next ten days:")
for i in range(len(predicted_prices)):
    print(f"{next_ten_dates[i]}: {predicted_prices[i]}")

plt.figure(figsize=(8, 6))
sns.histplot(data=Ddata, x='OpenOnly_as_PCT_of_DayRange', bins=20, kde=True)
plt.title('Histogram of Open Only Range as Percentage of Day Range')
plt.xlabel('Percentage')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
for window in windows:
    sns.lineplot(data=Ddata['ABSDayChange'].rolling(window=window).mean(), label=f'{window}-day Rolling Avg')
plt.title('Rolling Averages of Absolute Day Change')
plt.xlabel('Date')
plt.ylabel('Absolute Day Change')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(Ddata.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Ddata')
plt.show()

plt.figure(figsize=(10, 6))
for window in windows:
    sns.lineplot(data=Ddata['DayChange'].rolling(window=window).mean(), label=f'{window}-day Rolling Avg')
plt.title('Rolling Averages of Day Change')
plt.xlabel('Date')
plt.ylabel('Day Change')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=Ddata, x='DayChange', bins=20, kde=True)
plt.title('Histogram of Day Change')
plt.xlabel('Day Change')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=Ddata, x='DayRange', bins=20, kde=True)
plt.title('Histogram of Day Range')
plt.xlabel('Day Range')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=Ddata['ABSDayChange'], label='Absolute Day Change')
plt.title('Absolute Day Change Over Time')
plt.xlabel('Date')
plt.ylabel('Absolute Day Change')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=Ddata, x='OpenOnlyRange', bins=20, kde=True)
plt.title('Histogram of Open Only Range')
plt.xlabel('Open Only Range')
plt.ylabel('Frequency')
plt.show()

# Filter data for positive and negative DayChange
positive_changes = Ddata[Ddata['DayChange'] > 0]
negative_changes = Ddata[Ddata['DayChange'] < 0]

# Create subplots for positive and negative changes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for positive changes
sns.histplot(data=positive_changes, x='DayChange', bins=20, kde=True, ax=axes[0])
axes[0].set_title('Histogram of Positive Day Change')
axes[0].set_xlabel('Day Change')
axes[0].set_ylabel('Frequency')

# Plot for negative changes
sns.histplot(data=negative_changes, x='DayChange', bins=20, kde=True, ax=axes[1])
axes[1].set_title('Histogram of Negative Day Change')
axes[1].set_xlabel('Day Change')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Count negative and positive DayChange instances
negative_count = (Ddata['DayChange'] < 0).sum()
positive_count = (Ddata['DayChange'] > 0).sum()

# Create a DataFrame for counts
count_df = pd.DataFrame({'DayChange': ['Negative', 'Positive'], 'Count': [negative_count, positive_count]})

# Plot count of negative and positive DayChange values
plt.figure(figsize=(8, 6))
sns.barplot(data=count_df, x='DayChange', y='Count', palette='viridis')
plt.title('Count of Negative vs Positive DayChange')
plt.xlabel('Day Change')
plt.ylabel('Count')
plt.show()

# Filter data for positive DayChange
positive_changes = Ddata[Ddata['DayChange'] > 0]

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=positive_changes, x='OpenOnlyRange', y='DayChange', color='blue', alpha=0.7)
plt.title('Positive DayChange vs Open Only Range')
plt.xlabel('Open Only Range')
plt.ylabel('Day Change')
plt.show()

# Filter data for positive DayChange
positive_changes = Ddata[Ddata['DayChange'] > 0]

# Calculate the average Open Only Range for positive DayChange days
average_open_only_range = positive_changes['OpenOnlyRange'].mean()

# Create bar plot
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=['Positive DayChange'], y=[average_open_only_range], color='blue')
plt.title('Average Open Only Range on Positive DayChange Days')
plt.ylabel('Average Open Only Range')

# Annotate the bar plot with the value
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# Filter data for positive DayChange and Open Only Range > 105
positive_high_open_range = Ddata[(Ddata['DayChange'] > 0) & (Ddata['OpenOnlyRange'] > 105)]

# Count the number of positive days with Open Only Range > 105
count_positive_high_open_range = len(positive_high_open_range)

print("Count of positive days when Open Only Range is higher than 105:", count_positive_high_open_range)

# Filter data for negative DayChange and Open Only Range > 105
negative_high_open_range = Ddata[(Ddata['DayChange'] < 0) & (Ddata['OpenOnlyRange'] > 105)]

# Count the number of negative days with Open Only Range > 105
count_negative_high_open_range = len(negative_high_open_range)

print("Count of negative days when Open Only Range is higher than 105:", count_negative_high_open_range)

# Filter data for negative DayChange and Open Only Range < 105
negative_low_open_range = Ddata[(Ddata['DayChange'] < 0) & (Ddata['OpenOnlyRange'] < 105)]

# Count the number of negative days with Open Only Range < 105
count_negative_low_open_range = len(negative_low_open_range)

print("Count of negative days when Open Only Range is lower than 105:", count_negative_low_open_range)

# Filter data for positive DayChange
positive_days = Ddata[Ddata['DayChange'] > 0]

# Calculate the average Open Only Range value for positive days
average_open_only_range_positive_days = positive_days['OpenOnlyRange'].mean()

print("Average Open Only Range value for positive days:", average_open_only_range_positive_days)

# Filter data for negative DayChange
negative_days = Ddata[Ddata['DayChange'] < 0]

# Calculate the average Open Only Range for negative days
average_open_only_range_negative_days = negative_days['OpenOnlyRange'].mean()

print("Average Open Only Range for negative days:", average_open_only_range_negative_days)

# Filter data for positive DayChange and OpenOnlyRange < 105
positive_low_open_range = Ddata[(Ddata['DayChange'] > 0) & (Ddata['OpenOnlyRange'] < 105)]

# Count the number of positive days with OpenOnlyRange < 105
count_positive_low_open_range = len(positive_low_open_range)

print("Number of positive days with OpenOnlyRange under 105:", count_positive_low_open_range)

# Filter data for negative DayChange and OpenOnlyRange < 105
negative_low_open_range = Ddata[(Ddata['DayChange'] < 0) & (Ddata['OpenOnlyRange'] < 105)]

# Count the number of negative days with OpenOnlyRange < 105
count_negative_low_open_range = len(negative_low_open_range)

print("Number of negative days with OpenOnlyRange under 105:", count_negative_low_open_range)

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=Ddata, x='OpenOnlyRange', y='DayChange', color='blue', alpha=0.7)
plt.title('OpenOnlyRange vs DayChange')
plt.xlabel('OpenOnlyRange')
plt.ylabel('DayChange')
plt.show()

# Create joint plot
plt.figure(figsize=(8, 6))
sns.jointplot(data=Ddata, x='OpenOnlyRange', y='DayChange', kind='hex', color='blue')
plt.xlabel('OpenOnlyRange')
plt.ylabel('DayChange')
plt.show()