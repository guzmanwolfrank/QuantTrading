# Pivot Point Calculator 
by Wolfrank Guzman 

 # 

 Github: @guzmanwolfrank


![Pivot Points Header](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/f775a33a-8fe6-4cd9-9e45-bdfe2f8bd0b5)


## Project Overview 
 
The prime objective of this project is to make a tool useful to NQ 100 Emini Futures traders.  This tool helps calculate pivot points for the NQ 100 Emini contract. 
A seaborn output chart containing prices and pivot levels is also produced.  



### To view the project's Jupyter notebook, click [here](#)

### To view the project's Python code, click [here](#)

## Data Description

This Python script utilizes the yfinance library to fetch historical stock market data for a specified ticker over the last 30 days. The code focuses on the E-mini Nasdaq 100 (NQ) futures contract but can be adapted for other tickers as well.

#### Data Retrieval:
The default ticker is set to 'NQ=F'.
Historical data for the specified ticker is downloaded from Yahoo Finance for the last 30 days with daily intervals.

#
#### Pivot Points Calculation:

High, low, close, and open prices are extracted from the downloaded data and rounded to two decimal places.
Pivot points, support levels (S1, S2, S3), and resistance levels (R1, R2, R3) are calculated based on traditional pivot point analysis.
Data Presentation:

The calculated pivot points and relevant prices are organized into a Pandas DataFrame named pivot_data.
The last row of this DataFrame (excluding the most recent data) is extracted and printed for informational purposes.

#
#### Visualization:

The script uses Seaborn and Matplotlib to create a line plot depicting the closing prices along with the calculated pivot points, support, and resistance levels.
The y-axis labels are rotated by 90 degrees for better readability.
The resulting plot is displayed, providing a visual representation of the stock's price movement and key technical levels.

#
![pivoutput](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/adf77dda-d62b-4c69-93a8-0f69423cc967)



Note:

This script is designed to analyze and visualize the historical price movements of the specified stock, focusing on pivot points and support/resistance levels.
The code can be adapted for other stock tickers by modifying the initial user input.
Additional customization, such as choosing a different ticker or adjusting the date range, can be easily implemented based on specific requirements.


## Links: 

[yfinance](https://pypi.org/project/yfinance/)

## Project Objectives:
To make a useful utility for quick printing of daily intraday pivot points for the NQ 100 Emini CME Futures contract. 

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
   - **Description:** The project is licensed under the [MIT License](LICENSE), allowing users to freely use, modify, and distribute the code within the specified terms and conditions.

### 5. Version Control
   - **Repository:** [GitHub Repository](#)
   - **Description:** The project is version-controlled using Git. The repository provides a centralized location for tracking changes, collaborating with others, and maintaining a versioned history of the codebase.

### 6. Requirements File
   - **File:** `requirements.txt` (Optional)
   - **Description:** If applicable, a requirements file lists the Python libraries and their versions required to run the project successfully.



## Exploring the Data


This section provides insights into the data exploration process, outlining key steps and visualizations generated from the historical stock market data.

### 1. Data Retrieval

#### Ticker: NQ=F (Default)
- The project retrieves historical stock market data for the E-mini Nasdaq 100 (NQ) futures contract as the default ticker.
- Users can modify the ticker by providing input during script execution.

#### Date Range: Last 30 Days
- Data is fetched with daily intervals, covering the last 30 days.

### 2. Pivot Points and Technical Levels

#### Calculation
- Pivot points, support levels (S1, S2, S3), and resistance levels (R1, R2, R3) are calculated based on traditional pivot point analysis.

#### Data Organization
- The calculated pivot points and related prices are structured into a Pandas DataFrame named `pivot_data`.

#### Last Calculated Pivot Points
- The last row of pivot points (excluding the most recent data) is extracted and printed for informational purposes.

### 3. Data Visualization

#### Line Plot
- A line plot is generated using Seaborn and Matplotlib.
- It visualizes historical stock prices along with calculated pivot points, support, and resistance levels.
- The plot provides a clear representation of price movements and key technical levels over the specified date range.


![pivoutput](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/daad56b4-75e0-430c-b9fe-8edece707941)


#### Pivot Data Print
- Using .iloc method, a specific date is requested from the dataframe that shows that dates price and pivot levels. 
- It visualizes  stock prices along with calculated pivot points, support, and resistance levels.


![nqpivots](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/cceba7b9-f8d2-4150-b8d7-b6db54b332f5)


#### Dataframe 
- A pandas dataframe is displayed after runing the script. 
- The dataframe shows a table with Open, High, Low, and Close stock prices along with calculated pivot points, support, and resistance levels.

![pivdata](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/1d389fe4-14eb-4cff-ac73-4b5da6d37834)

## Findings 
Plotting pivots for NQ can help generate many backtest ideas for algorithms.  They also tend to flow and trend with the market.  They resemble Bollinger Bands once plotted
and can also generate spread and strangle/straddle spreads in futures options. 


## Conclusion

Plotting pivots can serve as a tremendous tool in analyzing futures markets and generating automated trade ideas. 
Another benefit is the ease with which you can plug the dataframe into an algorithmic backtest before deploying into the markets.

This project demonstrates the power of plotting pivots for uncovering potential trading opportunities in futures markets. 
With this visualization tool, traders can efficiently generate actionable trade ideas
and rapidly test them through backtesting, ultimately propelling informed decisions in live markets.

## Tech Stack 
yfinance==0.2.18  </br>
seaborn==0.12.2 </br>
pandas==2.0.3 <br/>
matplotlib==3.7.2 <br/>
Tableau==2023.3 <br/>



    Software: SQL, GoogleSheets, Python 3.11, VS Code, Jupyter Notebook, Tableau 
    Languages: SQL, Python
    Modules: Plotly, Pandas, SQLite3, Matplotlib

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
git clone https://github.com/your_username/your_project.git

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
