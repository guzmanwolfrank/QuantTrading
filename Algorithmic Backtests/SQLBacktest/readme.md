# USD/JPY Trading Strategy Backtest

Wolfrank Guzman: Github @guzmanwolfrank 

This project implements a simple trading strategy that backtests the performance of trading USD/JPY based on a specific strategy using historical data from Yahoo Finance. The strategy sells whenever the open price is under the 50-day moving average.

## Features

- Downloads historical USD/JPY data from Yahoo Finance.
- Calculates the 50-day moving average of the open price.
- Generates sell signals when the open price is under the 50-day moving average.
- Backtests the strategy and calculates profit and loss (P&L).
- Analyzes performance metrics using `quantstats` library.
- Saves the backtest results to a CSV file.

## Requirements

- Python 3.x
- yfinance
- quantstats

## Installation

You can install the required Python packages using pip:

```bash
pip install yfinance quantstats


## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/USD-JPY-Trading-Backtest.git


2. Navigate to the project directory: 

```bash
cd USD-JPY-Trading-Backtest

3. Run the Python script: 
```bash
python backtest.py


4. The backtest results will be saved to a CSV file named 
'backtest_results.csv.'


'''

## License

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.





