
# EURUSD Backtest 

This project is a moving average cross backtest for the EURUSD currency pair. The main objective is to beat the benchmark, a buy and hold strategy on EURUSD, and 
also to construct a simple enough algorithm for a beginner to edit and use. 



![download](https://github.com/guzmanwolfrank/Trading/assets/29739578/7f69d82c-c6c6-4888-98f4-bae893de9b33)

## Table of Contents 

I. [Introduction](https://github.com/guzmanwolfrank/Trading/tree/terra/Backtests/EURUSDMavg#introduction) <br />
II. Installation <br />
III. Tech Stack  <br />
IV. Project Details <br />
V. Issues, Challenges and Solutions <br />
VI.  Results <br />
VII. FAQ <br />
VIII. Links <br />
IX. License <br />
X. Related projects <br />
XI. Running Tests <br />
XII. Badges  <br />


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

### Challenge 1: <br />

One of the issues faced was selecting a proper data source. Certain other modules have unstable connections, deprecated software and are highly unavailable during peak times. 


### Solution 1: <br />
The solution was to find the best module with the most data integrity.  The data must be accurate, and reliable.  The best module for this backtest on currency was oandapyv20.  Oanda is a forex broker with excellent data which is reliable, accurate and highly available for clients.  



### Challenge 2:  <br />

    Coding a strategy on basic code that executes accurately while limiting slippage and price complexity.   

    Another challenge is downloading sufficient data while  limiting granularity in order to cap the size and latency of the data.    I wanted this to be simple, light and easy to customize.  

### Solution 2:  <br />

    I figured out to limit task bloating and complexity by using a basic signal strategy. 

    We will use a simple moving average cross along with a simplified trade entry system. 



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
## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.wolfrankguzman.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/wolfrank/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/wolfranknyc)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Related

Here are some related projects

[TQQQ Backtest](https://github.com/)


## Tech Stack

**Client:** React, Redux, TailwindCSS

**Server:** Node, Express


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


