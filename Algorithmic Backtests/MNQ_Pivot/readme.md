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

Successful Algorithm Implementation.
Seamlessly integrate the algorithm with the OANDA FX brokerage API.
Ensure accurate execution of trades based on generated signals.
Implement robust error handling and risk management mechanisms.

Effective Trading Performance:

Generate consistent and profitable trading results in live market conditions.
Evaluate and optimize the algorithm's performance through backtesting and forward testing.
Fine-tune signal generation and trade execution parameters for optimal outcomes.
Comprehensive Monitoring and Analysis:

Develop a system to track and analyze the algorithm's performance in real-time.
Generate detailed performance reports, including profit and loss statements, drawdowns, win rates, and other key metrics.
Identify areas for improvement and make necessary adjustments to the algorithm.
Continuous Evaluation and Improvement:

Regularly review and assess the algorithm's performance in response to changing market conditions.
Adapt and refine the algorithm as needed to maintain its effectiveness.
Explore potential enhancements, such as incorporating additional technical indicators or risk management strategies.

# Initial Questions 
1. Is the strategy profitable in a backtest? Does it beat the benchmark strategy?
2. What are the return metrics?  Sharpe ratio? Sortino?  Max Drawdown? 


# Findings 

**Profitability**: The algorithm achieved a positive overall profit (XXXXX%) over the backtesting period. This suggests that the Pivot Strategy was able to successfully identify profitable trading opportunities.<br/>
Win Rate: The algorithm maintained a win rate of [XXXXXX%] over the backtesting period, indicating that it captured a significant portion of positive trades.<br/>
Risk Management: The implemented risk management measures effectively limited drawdowns and protected the capital.
<br/>
**Areas for Improvement**
<br/>
Drawdown Optimization: While drawdowns were controlled, further optimization could potentially reduce their extent without significantly impacting profitability.
<br/>
Signal Refinement: Analyzing false positives and missed opportunities could lead to refining the signal generation rules for enhanced accuracy.
<br/>
Market Adaptability: Evaluating the algorithm's performance across different market conditions (trending, ranging, volatile) can reveal potential weaknesses and suggest adaptation strategies.

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

