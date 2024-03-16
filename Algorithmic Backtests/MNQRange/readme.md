MNQ Range 
#
### by Wolfrank Guzman 
@guzmanwolfrank : Github 


![mnqchart](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/55ad6326-bd1b-4fa5-8c05-af5a98800868)


# Project Overview 

This machine learning project aims to predict future closing prices after measuring MNQ price ranges in the first 45 minutes of the Equities Open along with the Daily Price Change and Day's Range from High to Low.
We will be looking for correlations using Seaborn visualizations, Pandas Dataframes, sci-kit module and YFinance downloaded data.

### To view the project's Jupyter notebook, click [here](#)

# Data Description

This section describes the key columns within the dataset used by the backtest and machine learning module (sci-kit).

### Price Data:

Open: The opening price of the MNQ Futures contract for a given period. <br/>
High: The highest price reached by MNQ during that period.<br/>
Low: The lowest price reached by MNQ during that period.<br/>
Close: The price at which MNQ closed for the period. This price is used by the algorithm to determine the main signal based on its proximity to the pivot point.<br/>
Adj Close: The adjusted closing price, taking into account dividend distributions and stock splits.<br/>

### Pivot Points:
PP: The pivot point, calculated as the average of the previous day's high, low, and close prices.<br/>

### Algorithm-Specific Data:
ABSDayChange:  This column shows the absolute value of the Day's Change which is the Close - Open.  
DayChange:  The positive or negative value of the Day's change which is the Close - Open. 
OpenOnlyRange:   This column shows the value of the min and max value in the first 45 minutes of the Equities Market Open at 9:30. 
DayRange:   This value contains the distance from the Day's High to Low or vice versa.

## Project Objectives 


To develop and implement a backtest for an algorithm utilizing machine learning in order to predict future closing prices along with analyzing price range data on MNQ futures. 

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


# Exploring the Data
This section provides insights into the data exploration process, outlining key steps and visualizations generated from the historical market data.

### 1. Data Retrieval

#### Ticker: MNQ (Default)
- The project retrieves historical stock market data for the MNQ as the default ticker.
- Users can modify the ticker directly into the notebook.

#### Date Range: 
<br/>
granularity = 'D'  # Daily data 
<br/>
start_date = "2024-03-01"
<br/>
end_date = "2024-03-11"
<br/>

- Data is fetched with daily intervals, covering the last year.

<br/>

### 2. Pivot Points and Technical Levels

#### Calculation
- Pivot points, are calculated based on traditional pivot point analysis.

#### Data Organization
- The calculated pivot points and related prices are structured into a Pandas DataFrame named `data`.

#### Last Calculated Pivot Points
- The appropiate entry of pivot points (excluding the most recent data) is extracted and printed for informational purposes.

### 3. Data Visualization

#### QuantStats HTML Tearsheet 
- QuantStats module prints results and risk metrics to an external webpage.  You can save it locally or to your online Github repository. 



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
python XXXXXX.py

```


## Badges 

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## License 
[MIT](https://choosealicense.com/licenses/mit/)

