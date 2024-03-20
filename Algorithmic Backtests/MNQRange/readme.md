# MNQ Range 
#
### by Wolfrank Guzman 
@guzmanwolfrank : Github 


![mnqchart](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/55ad6326-bd1b-4fa5-8c05-af5a98800868)


# Project Overview

This project analyzes Micro E-mini NASDAQ-100 Futures (MNQ) using Python libraries such as `yfinance`, `pandas`, `seaborn`, and `scikit-learn`. It includes data retrieval, preprocessing, visualization, and predictive modeling.


### To view the project's Jupyter notebook, click [here](https://github.com/guzmanwolfrank/QuantTrading/blob/main/Algorithmic%20Backtests/MNQRange/MNQrange.ipynb)


## Data Description

The data consists of intraday and historical price information for MNQ fetched using the `yfinance` library. Intraday data covers the last 60 days with 15-minute intervals, while historical data spans one year.

## Project Objectives

The objectives of this project are to:

- Analyze intraday and historical price data for MNQ.
- Calculate summary statistics and rolling averages for specific time intervals.
- Visualize trends and rolling averages using line plots.
- Predict future closing prices using linear regression.

## Project Deliverables

The project delivers:

- Insights into intraday and historical price movements for MNQ.
- Visualizations depicting trends and rolling averages.
- Predicted closing prices for the next ten days.

## Goals

- Gain insights into MNQ price movements during specific time intervals.
- Understand the relationship between different price metrics and rolling averages.
- Predict future closing prices for MNQ using linear regression.

## Initial Questions

1. What are the average opening ranges for different windows of days?
2. What percentage of the day range is the open-only range?
3. Can we use machine learning to predict futures prices? 

## Findings

- Average opening ranges for different window sizes show variations over time.
- The percentage of the day range represented by the open-only range varies across instances.
- Rolling averages highlight trends and fluctuations in absolute day change, open-only range, and day range.


The histogram analysis of the open-only range as a percentage of the day range provided valuable insights. Notably, there were **33 instances** where `'OpenOnly_as_PCT_of_DayRange'` exceeded 35%, indicating significant intraday volatility within the MNQ futures contract. Moreover, in **18 instances**, the open-only range surpassed 50% of the day range, highlighting pronounced fluctuations early in the trading session.

The highest frequency count for percentage was **40%**, observed in more than 8 instances, suggesting that the open range typically represents a substantial portion of intraday volatility.

Traders seeking heightened volatility may find it advantageous to trade within the first 45 minutes of the trading session. However, this strategy is not recommended for novices due to its potential complexities.

Additionally, our analysis identified that the most frequent open range value is **100 points**, occurring over 14 times out of the 50-day instances.

Furthermore, our brief examination revealed that negative days tend to exhibit higher volatility in terms of overall intraday range.

Understanding these findings can aid in structuring a backtest that capitalizes on the opening range volatility while assessing abnormal price movements outside of our custom-calculated averages.

![output](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/81fa2477-585e-40ba-9ed5-08cf99f1af58)

![output2](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/5a574666-7f09-4d1c-b6a7-392f58d4168d)


## Conclusion

The analysis provides valuable insights into MNQ price dynamics, enabling better understanding and prediction of future price movements. By leveraging intraday and historical data, along with advanced analytics techniques, this project enhances decision-making in trading and investment strategies.

## Tech Stack

- Python
- Libraries: yfinance, pandas, seaborn, matplotlib, numpy, scikit-learn

## Dependencies

Ensure you have the following Python libraries installed:

yfinance, pandas, seaborn, matplotlib, numpy, scikit-learn


## License

This project is licensed under the MIT License.
