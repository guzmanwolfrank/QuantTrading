#  Quant Trading
A collection of algorithmic backtests and trading tools written in python.

N.B.:  All backtests are simple and simply looking for return criteria to be fulfilled.  Complex orders, size parlays, and defined stops are calculated in algorithm for deployment. 
All backtests so far are using Open as the entry and Close as the exit as long as the signal is triggered or present.  
<br>
Data:  Gathered from Brokerage API using Oanda Forex Brokers, YFinance Python Module 
<br>

Adjustments:   
The idea behind these backtests is to look for exponential returns to then refine the system and iterate across various market conditions before adding complexity and retesting.
After retesting, and adjusting the dataframe with:  Commissions, Slippage (*orders estimated to open price), Order size(Size parlaying can dramatically increase losses and profits, if done correctly, scaling can have exponential returns).
<br> 
Add slippage and look at the average open range in order to simulate entering a market order at the open after the signal is present.  By analyzing the range of the first 2 minutes of trading, one can estimate the slippage on entry price off this system. 
**Work in progress

#### [Algorithmic Backtests](https://github.com/guzmanwolfrank/QuantTrading/tree/main/Algorithmic%20Backtests): This folder contains algorithmic backtest programs and projects.     

![299770993-136bf05a-9b74-4ee7-8951-b28983e4e3e5](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/1a35e249-dcd5-45ed-8b8e-326ebc8fefd4)


    Software: Python 3.11, VS Code, Jupyter Notebook
    Languages:  Python
    Modules: Seaborn, Pandas, Yfinance, Matplotlib, numpy, Plotly, QuantStats, datetime


#### [Tools ](https://github.com/guzmanwolfrank/QuantTrading/tree/main/Tools): This portfolio contains tools and mini projects that enhance trading data systems.    

![nqpivots](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/29695637-1150-4634-8c11-51fba32f7086)


    Software: Python 3.11, VS Code, Jupyter Notebook
    Languages:  Python
    Modules: Seaborn, Pandas, Yfinance, Matplotlib


