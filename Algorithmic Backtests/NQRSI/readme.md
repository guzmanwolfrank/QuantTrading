# Bollinger Bands and RSI Trading Strategy

This Python script implements a trading strategy based on Bollinger Bands and the Relative Strength Index (RSI). The strategy is backtested using historical data for NQ Futures obtained from Yahoo Finance. It calculates various metrics such as profits, returns, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.

## Strategy Overview

The strategy follows these steps:

1. **Data Acquisition**: Historical data for NQ Futures is downloaded using the Yahoo Finance API.

2. **Bollinger Bands Calculation**: Bollinger Bands are calculated based on the closing prices, with default parameters of a 20-day window and 2 standard deviations.

3. **RSI Calculation**: The Relative Strength Index (RSI) is calculated to identify overbought and oversold conditions in the market.

4. **Backtest Strategy**: Based on Bollinger Bands and RSI signals, the strategy determines whether to buy, sell, or hold positions.

5. **Profit Calculation**: Profits are calculated based on the points gained or lost from each trade, considering commissions.

6. **Performance Metrics**: Various performance metrics such as total profit, total returns, annualized returns, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown are calculated.

7. **Visualization**: A snapshot of the strategy's performance is generated using QuantStats.

## Results

| Metric                  | Benchmark    | Strategy    |
|-------------------------|--------------|-------------|
| Start Period            | 2020-01-03   | 2020-01-03  |
| End Period              | 2023-12-29   | 2023-12-29  |
| Risk-Free Rate          | 0.0%         | 0.0%        |
| ...                     | ...          | ...         |

[Strategy Visualization]
via Matplotlib
Total Profit: 25036.311386718742
Total Returns: 0.7145818548448379
Annualized Returns: 0.14472262986420348
Sharpe Ratio: 1.8685647200423676
Sortino Ratio: 3.043519207063549
Maximum Drawdown: -0.17419477215633627

![image](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/21c7af35-4a87-4bbf-8704-432e42d27402)


## Conclusion

In conclusion, the Bollinger Bands and RSI trading strategy presented in this Python script demonstrates robust performance metrics and promising returns based on backtesting with NQ Futures data. By leveraging technical indicators such as Bollinger Bands and the Relative Strength Index, the strategy aims to capitalize on market momentum and identify optimal entry and exit points for trades.
<br>The comprehensive analysis of performance metrics including Sharpe Ratio, Sortino Ratio, and Maximum Drawdown provides valuable insights into the strategy's risk-adjusted returns and downside protection.
<br>
 Additionally, the visualization generated using QuantStats offers a clear snapshot of the strategy's performance over the backtesting period. Overall, the strategy showcases its potential to generate consistent profits and outperform benchmark indices, making it a compelling option for traders seeking to capitalize on market opportunities. 
 
 <br>Further refinement and optimization of the strategy parameters could potentially enhance its effectiveness and adaptability across different market conditions.

## Running the Script

To run the script:

1. Ensure you have Python installed on your system along with the necessary libraries (`yfinance`, `pandas`, `quantstats`).

2. Open the script in a Python environment or run it from the command line.

3. The script will download historical data, perform calculations, and output the results including performance metrics and a snapshot of the performance.

## Dependencies

- [yfinance](https://pypi.org/project/yfinance/): A Python library to download historical market data from Yahoo Finance.
- [pandas](https://pandas.pydata.org/): A powerful data manipulation library in Python.
- [quantstats](https://github.com/ranaroussi/quantstats): A Python library for analyzing financial performance.

## Author

[Wolfrank Guzman/@guzmanwolfrank: github]

## License

This project is licensed under the [MIT License](LICENSE).
