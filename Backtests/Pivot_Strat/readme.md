

# EUR USD Pivot Strategy Backtest 
#
### by Wolfrank Guzman 
@guzmanwolfrank : Github 


![qimage](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/136bf05a-9b74-4ee7-8951-b28983e4e3e5)


# Project Overview 

This project creates an algorithm based on the [EURUSD Pivot Strategy backtest](https://github.com/guzmanwolfrank/QuantTrading/tree/main/Backtests/Pivot_Strat). 
The main signal is generated depending on the price of the close relative to the intraday pivot point.

his bot is designed for use on OANDA FX brokerage API.  



### To view the project's Jupyter notebook, click [here](#)

### To view the project's Tableau Dashboard, click [here](#)

# Data Description

Open	
High	
Low	
Close
Adj Close
Volume	
PP
R1
S1
Signal
Returns
Profit
Balance



# Links

oanda API
pandas 


## Project Objectives 


To develop and implement an automated trading algorithm on the OANDA FX brokerage platform, utilizing the EURUSD Pivot Strategy to generate buy and sell signals based on the price of the close relative to the intraday pivot point.



## Project Deliverables:  
This section outlines the key deliverables and artifacts associated with the project, providing a comprehensive overview of what users can expect from the codebase.

### 1. Python Script
   - **File:** `wolfpivots.py`
   - **Description:** The main Python script that performs the following tasks:
      - Downloads historical stock market data for a specified ticker.
      - Calculates pivot points, support levels (S1, S2, S3), and resistance levels (R1, R2, R3).
      - Visualizes historical stock price movements, pivot points, and technical levels using Seaborn and Matplotlib.
      - Displays relevant information, including the last calculated pivot points.

### 2. Jupyter Notebook 
   - **File:** `wolfpivots.ipynb`
   - **Description:** Jupyter notebook that performs the following tasks:
      - Downloads historical stock market data for a specified ticker.
      - Calculates pivot points, support levels (S1, S2, S3), and resistance levels (R1, R2, R3).
      - Visualizes historical stock price movements, pivot points, and technical levels using Seaborn and Matplotlib.
      - Displays relevant information, including the last calculated pivot points.


### 3. README
   - **File:** `README.md`
   - **Description:** This document provides comprehensive information about the project, including:
      - Overview of the project's purpose and functionality.
      - Installation instructions and dependencies.
      - Explanation of tools and technologies used.
      - Details on how to run the Python script and contribute to the project.
      - Licensing information and guidelines for contributions.

### 4. License
   - **File:** `LICENSE`
   - **Description:** The project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/), allowing users to freely use, modify, and distribute the code within the specified terms and conditions.

### 5. Version Control
   - **Repository:** [GitHub Repository](https://github.com/guzmanwolfrank/QuantTrading/tree/main/Tools/Wolf%20Pivot%20Calculator)
   - **Description:** The project is version-controlled using Git. The repository provides a centralized location for tracking changes, collaborating with others, and maintaining a versioned history of the codebase.

### 6. Requirements File
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

# Exploring the Data

# Visualizations 

# Findings 

# Conclusion

## Tech Stack 
yfinance==0.2.18  </br>
seaborn==0.12.2 </br>
pandas==2.0.3 <br/>
matplotlib==3.7.2 <br/>
Tableau==2023.3 <br/>



    Software: Python 3.11, VS Code, Jupyter Notebook
    Languages:  Python
    Modules: Seaborn, Pandas, Yfinance, Matplotlib

## Project Structure

- **README:**
  - **Description:** This document provides an overview of the project, its purpose, and instructions for running and understanding the code.

- **Python Script:**
  - **Description:** The main script written in Python that fetches financial data, calculates pivot points, and visualizes the historical stock price movements.

## Dependencies

- Ensure that the required Python libraries are installed. You can install them using the following command:


```bash
pip install yfinance pandas seaborn matplotlib

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
python wolfpivots.py

```


## Badges 

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## License 
[MIT](https://choosealicense.com/licenses/mit/)

