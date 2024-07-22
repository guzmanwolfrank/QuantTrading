# OpenStrat 
@guzmanwolfrank



A simple strategy for creating buy and sell positions within the first 15 minutes of trading. 

Strategy 
    Open a short order when volume is high and the first fifteen minutes of trading is under the intraday pivot point and the close of the prior day.

    Open a buy order when volume is high and the first 15 minutes of trading is over the previous day high. 

    Open a buy order when the ador average daily opening range is higher than the current daily opening range.  this is the range of high to low in the first 15 minutes of trading. 

    open a sell order when the ador is lower than the current daily opening range and the price is lower than the previous days close

    if a forex trade use a 10 pip stop loss and 25 pip win close order 

    run on the 15 min candlestick chart

    if a microfutures trade use a 24 pt stop loss and 45 point profit stop 


# commands for ai

Create a backtest on usdjpy and mnqu4, using Quantstats, Yfinance, Pandas that follows the following strategy:

Open a short order when volume is high and the first fifteen minutes of trading is under the intraday pivot point and the close of the prior day.

    Open a buy order when volume is high and the first 15 minutes of trading is over the previous day high. 

    Open a buy order when the ador average daily opening range is higher than the current daily opening range.  this is the range of high to low in the first 15 minutes of trading. 

    open a sell order when the ador is lower than the current daily opening range and the price is lower than the previous days close

    if a forex trade use a 10 pip stop loss and 25 pip win close order 

    run on the 15 min candlestick chart

    if a microfutures trade use a 24 pt stop loss and 45 point profit stop 