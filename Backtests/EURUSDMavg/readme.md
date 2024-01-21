

# EURUSD Backtest 

This project is a moving average cross backtest for the EURUSD currency pair. The main objective is to beat the benchmark, a buy and hold strategy on EURUSD, and 
also to construct a simple enough algorithm for a beginner to edit and use. 


![download](https://github.com/guzmanwolfrank/Trading/assets/29739578/7f69d82c-c6c6-4888-98f4-bae893de9b33)

### To view the project's Jupyter notebook, click [here](https://github.com/guzmanwolfrank/QuantTrading/blob/QuantTrading/Backtests/EURUSDMavg/Strat.ipynb)

### To view the project's Tableau Dashboard, click [here](https://public.tableau.com/app/profile/wolfrank.guzman/viz/EURUSD_17057990902740/Dashboard1?publish=yes)

![tableaustats](https://github.com/guzmanwolfrank/QuantTrading/assets/29739578/15bb574a-62ae-4f8c-bb0c-b84d52beb84d)

## Table of Contents 

I. [Introduction](https://github.com/guzmanwolfrank/Trading/tree/terra/Backtests/EURUSDMavg#introduction) <br />
II. [Installation](https://github.com/guzmanwolfrank/Trading/tree/working/Backtests/EURUSDMavg#installation) <br />
III. [Project Details](https://github.com/guzmanwolfrank/Trading/tree/working/Backtests/EURUSDMavg#project-details)  <br />
IV. [Challenges and Solutions](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#challenges-and-solutions) <br />
V.  [Results](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#results) <br />
VI. [Conclusion](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#conclusion) <br />
VII. [Links](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#links) <br />
VIII. [License](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#license) <br />
IX. [Related projects](https://github.com/guzmanwolfrank/Trading/tree/main/Backtests/EURUSDMavg#) <br />
X. [Tech Stacks](https://github.com/guzmanwolfrank/Trading/tree/working/Backtests/EURUSDMavg#tech-stack) <br />
XI. [Running Tests](https://github.com/guzmanwolfrank/Trading/tree/working/Backtests/EURUSDMavg#running-tests)  <br />
XI. [Badges](https://github.com/guzmanwolfrank/Trading/tree/working/Backtests/EURUSDMavg#badges)  <br />


## Introduction 

This project is intended to help a beginner start working on algorithms and backtests on simple quant strategies.  
We will be using a Forex Brokerage account as a Data Provider and its Python module as well. 


## Installation

1. To install and run this program, you must have a jupyter notebook running on a python kernel. <br />

2.  Download the Strat.ipynb file into a root folder where you will also download the necessary python modules. <br />

3.  Open the file in a Jupyter notebook.  <br />

4. Pip install the modules for this 
project into your root folder using the pip install command as shown:

![carbon (2)](https://github.com/guzmanwolfrank/Trading/assets/29739578/f0f09919-20f8-4e3f-8b1c-58deb1e296e2)
<br />


5.  Run the program in the Jupyter notebook.  Click on the Run All button after selecting a Python 3.9 kernel.  <br />


## Project Details

This project is intended to help a beginner start working on algorithms and backtests on simple quant strategies.  
We will be using a Forex Brokerage account as a Data Provider and its Python module as well. 

The strategy consists of buying on a signal that is triggered when the 10 day moving average is greater than the day's open. 
The transaction then occurs on the following day's open and close. 
The program will then record results into a dataframe which can then be visualized with seaborn and other python modules. 


Backtest and Transaction Details: 
#
        Currency Pair: EUR/USD 
        Account Balance: $ 10,000
        Leverage : 50X
        Average Order Size: 50,000
        Order Entry: Buy Market at Open
        Order Exit: Sell Market at Close
        Margin Used: $ 1,000
        Margin: 2%


We will run multiple iterations of the backtest through set dates in order to analyze the data and see if the strategy is worth pursuing. 


## Challenges and Solutions

**Challenge 1**: <br />

One of the issues faced was selecting a proper data source. There are a myriad of data sources and modules but certain modules have unstable connections, deprecated software and are highly unavailable during peak times. 
This can lead to incomplete, inaccurate or corrupted data.  

**Solution 1**: <br />
The solution was to find the best module with the most data integrity.  The data must be accurate, and reliable.  The best module for this backtest on currency was oandapyv20.  Oanda is a forex broker with excellent data which is reliable, accurate and highly available for clients.  



**Challenge 2**:  <br />

Coding a strategy on basic code that executes accurately while limiting slippage and price complexity.   

Another challenge is downloading sufficient data while  limiting granularity in order to cap the size and latency of the data.    I wanted this to be simple, light and easy to customize.  

**Solution 2**:  <br />

I figured out to limit task bloating and complexity by using a basic signal strategy. We will use a simple moving average cross along with a simplified trade entry system. 
A -1 signal equals a sell, where a 1 signals a buy.  I eliminated the sell signals in order to simplify transactions and make them buy orders at market.
Another simplification was the use of market orders at specific times like the open and close.  

I limited granularity to the Day Chart on EURUSD in order to limit transaction frequency while being congruent to the exit strategy.  

![signal](https://github.com/guzmanwolfrank/Trading/assets/29739578/18866b8e-9922-4e50-848c-4d159ec36e28)


**Challenge 3**:  <br />

The next challenge was in building a backtest was designing a simple risk mitigation system or management within the simplicity of the strategy. 
I wanted to contain any possible losses to as tight an amount as possible while letting the market play out in an intraday system. 

**Solution 3**:   <br />

The solution to this challenge was to close the transactions on the same day they were opened by closing on the close at 4pm. 
This way, we can limit our losses to the average daily change from open to close on the currency pair.  

![carbon (3)](https://github.com/guzmanwolfrank/Trading/assets/29739578/564b0718-0035-4391-8aed-f9fd253bb799)


## Results 

Some of the questions we looked to answer were:

        1.  What were the results of the strategy?
        2.  What was the drawdown?
        3.  How did the benchmark perform?
        4.  What were the risk metrics? 
        5.  What were the average profits? Losses?
        6.  What was the win/loss count?
        7.  What is the average daily range? 
        8.  What is the average daily change?
        9.  What is the rate of return on the strategy?
        10. What were the most and least profitable trades? 


![resultchart](https://github.com/guzmanwolfrank/Trading/assets/29739578/24b29768-82bc-4871-99dd-79214b5c7c0a)

## Conclusion 

<br/>
Backtest Conclusion: 

High-performing strategy outpaces benchmark with manageable risk.

<br/>
This backtest demonstrates a highly successful trading strategy with impressive returns of 79% and a Sharpe Ratio of 1.74. While drawdowns occurred (maximum of 17,916), risk metrics were considered and the strategy significantly outperformed the benchmark, doubling its final balance. Average daily change and range were minimal, suggesting consistency and control. Overall, this strategy presents a promising option with strong performance and controlled risk.
<br/>

<hr/> 

Key takeaways:

High returns: 79% average return, 101.5% rate of return.
Strong risk management: Drawdowns monitored, average drawdown manageable.
Outperformance: Strategy significantly beat the benchmark.
Consistency: Low average daily change and range.

<hr/> 


<br/>
Further investigation:

Analyze win/loss ratio and trade distribution for deeper insights.
Explore diversification potential to mitigate risk.
Test strategy under different market conditions to assess robustness.
<br/>

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.wolfrankguzman.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/wolfrank/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/wolfranknyc)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Related Projects

Here are some related projects

[TQQQ Backtest](https://github.com/)


## Tech Stack

**Language**: python==3.11.3
 
**Modules**:
oandapyV20==0.6.3 <br />
pandas==1.5.0 <br />
numpy==1.23.3 <br />
matplotlib==3.6.0 <br />
seaborn==0.12.0 <br />
DateTime==4.7 <br />

**Framework**:  Jupyter, Jupyter Notebook


## Running Tests

To run tests, run the following command

```bash
  npm run test
```


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


