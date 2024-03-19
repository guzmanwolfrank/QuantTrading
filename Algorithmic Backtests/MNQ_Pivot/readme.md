# MNQ Pivot Strategy Backtest 
#
### by Wolfrank Guzman 
@guzmanwolfrank : Github 


![mnqchart](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/55ad6326-bd1b-4fa5-8c05-af5a98800868)


# Project Overview 

This project creates a backtest based on the MNQ Pivot point prices.  

### To view the project's Jupyter notebook, click [here](https://github.com/guzmanwolfrank/QuantTrading/blob/main/Algorithmic%20Backtests/MNQ_Pivot/mnqpivotsbt.ipynb)

# Data Description

This Python script utilizes various financial libraries to analyze intraday data of a specific ticker symbol, calculating pivot points, moving averages, and profitability metrics. Below is a breakdown of the data processing steps:

**Fetching Data**: Intraday data for the last 30 days with 15-minute intervals is fetched using the yfinance library for the ticker symbol "MNQ=F" (Micro E-mini Nasdaq 100 Futures).

**Calculation of Pivot Points**: Pivot points are calculated for each 15-minute interval using the high, low, and close prices. Support and resistance levels are derived from these pivot points using specific formulas.

**Moving Average**: A 10-period moving average is calculated for the close price to smooth out fluctuations.

**Signal Generation**: A signal column is created where a value of 1 is assigned whenever the pivot point is greater than the opening price, indicating a potential bullish signal.

**Profit Calculation**: Profit is calculated based on the change in price, with each point of change in the "MNQ" futures contract representing a profit of 2 dollars. Positions are opened at the opening price and closed at the closing price of the same day.

**Performance Metrics**: Performance metrics such as cumulative profit, balance, and returns are calculated using the quantstats library.

**Initial Balance**: An initial balance of $10,000 is considered for the trading simulation.

**Display**: The processed data including pivot points, moving averages, signals, and profitability metrics are displayed for analysis.

**Performance Report**: Performance metrics are displayed using the quantstats library, offering insights into the trading strategy's effectiveness compared to a benchmark (in this case, the "MNQ=F" ticker).

**HTML Tearsheet (Optional)**: An HTML tearsheet can be generated to provide a visual summary of the trading strategy's performance compared to the benchmark.

Note: The profitability metrics and performance evaluation are based on simulated trading and should not be considered as financial advice.








## Project Objectives 


To develop and implement a backtest for an algorithm utilizing a Pivot Strategy to generate buy signals based on the price of the Open relative to the intraday pivot point.



## Project Deliverables:  
This section outlines the key deliverables and artifacts associated with the project, providing a comprehensive overview of what users can expect from the codebase.


### 1. Jupyter Notebook & Python Script
   - **Files:** `mnqpivotsbt.ipynb`, 'mnqpivotsbt.py'
   - **Description:** Jupyter notebook and python script that perform the following tasks:
      - Downloads historical stock market data for a specified ticker.
      - Calculates pivot points.
      - Deploys strategy and records historical stock price movements, pivot points, and technical levels.
      - Displays relevant information, including the last calculated pivot points and QuantStats module output.


### 2. README
   - **File:** `README.md`
   - **Description:** This document provides comprehensive information about the project, including:
      - Overview of the project's purpose and functionality.
      - Installation instructions and dependencies.
      - Explanation of tools and technologies used.
      - Details on how to run the Python script and contribute to the project.
      - Licensing information and guidelines for contributions.

### 3. License
   - **File:** `LICENSE`
   - **Description:** The project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/), allowing users to freely use, modify, and distribute the code within the specified terms and conditions.

### 4. Version Control
   - **Repository:** [GitHub Repository](#)
   - **Description:** The project is version-controlled using Git. The repository provides a centralized location for tracking changes, collaborating with others, and maintaining a versioned history of the codebase.

### 5. Requirements File
   - **File:** `requirements.txt` 

# Goals

The primary goal of this project is to provide a Python script that facilitates the analysis of intraday price data for the "MNQ=F" ticker symbol, focusing on the following objectives:

1. **Data Processing**: Develop a robust data processing pipeline to fetch, clean, and analyze intraday price data using Python libraries such as `yfinance` and `pandas`.

2. **Technical Analysis**: Implement common technical analysis techniques, including the calculation of pivot points, support/resistance levels, and moving averages, to identify potential trading opportunities and trends in the market.

3. **Signal Generation**: Generate signals based on predefined criteria, such as pivot point comparisons with opening prices, to assist traders in making informed decisions regarding entry and exit points in the market.

4. **Profitability Metrics**: Calculate profitability metrics, including profit per trade, cumulative profit, and returns, to evaluate the performance of the trading strategy over time and assess its effectiveness in generating profits.

5. **Performance Evaluation**: Provide comprehensive performance evaluation tools using the `quantstats` library to analyze the strategy's performance relative to a benchmark, offering insights into its profitability and risk-adjusted returns.

6. **Accessibility and Customization**: Ensure that the script is well-documented, easy to understand, and customizable, allowing users to modify parameters and adapt the analysis to their specific requirements and preferences.

7. **Educational Resource**: Serve as an educational resource for individuals interested in learning about intraday trading strategies, technical analysis techniques, and quantitative finance concepts through practical examples and hands-on experimentation.

By achieving these goals, this project aims to empower traders and analysts with the tools and insights necessary to make informed decisions in the dynamic and fast-paced world of intraday trading.


# Initial Questions 
1. Is the strategy profitable in a backtest? Does it beat the benchmark strategy?
2. What are the return metrics?  Sharpe ratio? Sortino?  Max Drawdown? 


# Findings 


### Backtest Profitability

The conducted backtest analysis reveals that the strategy exhibits profitability over the specified time period. By leveraging technical analysis indicators such as pivot points and moving averages, the strategy identifies potential trading opportunities effectively, resulting in positive returns.

### Benchmark Comparison

In comparison to the benchmark strategy, the developed trading strategy demonstrates promising performance. While further analysis is warranted, initial findings suggest that the strategy outperforms the benchmark in terms of profitability and risk-adjusted returns.

### Return Metrics

- **Sharpe Ratio**: The Sharpe ratio measures the risk-adjusted return of the strategy. Initial calculations indicate a favorable Sharpe ratio, suggesting that the strategy generates returns while effectively managing risk.
  
- **Sortino Ratio**: The Sortino ratio assesses the risk-adjusted return, focusing solely on downside risk. Analysis indicates a favorable Sortino ratio, indicating that the strategy delivers consistent returns while minimizing downside volatility.
  
- **Max Drawdown**: The maximum drawdown quantifies the largest peak-to-trough decline experienced by the strategy during the backtest period. Evaluation of the maximum drawdown suggests acceptable risk levels, with the strategy demonstrating resilience during adverse market conditions.

Further exploration and fine-tuning of the strategy parameters may offer opportunities for enhancing performance metrics and refining risk management strategies.


# Conclusion

The backtest demonstrates the potential of the EURJPY Pivot Strategy to generate profitable trading signals. However, further optimization and testing are recommended to address the identified areas for improvement and ensure robustness in live market conditions.

Disclaimer: These findings are based solely on the backtesting results and may not necessarily translate to consistent profitability in live trading.



## Tech Stack 
pandas==2.0.3 <br/>
QuantStats==0.0.62 <br/>


    Software: Python 3.11, VS Code, Jupyter Notebook
    Languages:  Python
    Modules: YFinance, Pandas, QuantStats

## Project Structure

- **README:**
  - **Description:** This document provides an overview of the project, its purpose, and instructions for running and understanding the code.

- **Python Script:**
  - **Description:** The main script written in Python that fetches financial data, calculates pivot points, and visualizes the historical stock price movements.

## Dependencies

- Ensure that the required Python libraries are installed. You can install them using the following command:


```bash
pip install yfinance pandas quantstats

#

### Getting Started

Clone the Repository:
$bash
git clone (git url)

Install Dependencies:
$bash
pip install -r requirements.txt


Run the Script:
$bash
python mnqpivotsbt.py

```


## Badges 

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## License 
[MIT](https://choosealicense.com/licenses/mit/)

