# Wolf RSI Filter [NQ100 Stock Screener]

## Objective
The objective of this program is to build a stock screener that identifies NQ100 stocks exhibiting specific criteria:
- The Relative Strength Index (RSI) should be below 30.
- Volume should show a consistent increase over a minimum span of 5 days.
- The volume must grow incrementally by at least 10% each day.

## Features
- **RSI Filter:** The program utilizes the RSI indicator to filter out stocks with an RSI below 30.
- **Volume Growth:** It analyzes the volume data of each stock to ensure it increases over the specified span, with a minimum growth rate of 10% each day.
- **Customizable Parameters:** Users can adjust parameters such as the RSI threshold, the minimum span of days, and the growth rate of volume.

## How to Use
1. **Input Data:** Provide the program with historical stock data for NQ100 stocks.
2. **Set Parameters:** Adjust the parameters according to your preferences, such as the RSI threshold and the minimum span of days.
3. **Run the Program:** Execute the program to screen the stocks based on the defined criteria.
4. **Review Results:** Examine the list of NQ100 stocks that meet the specified conditions.

## Requirements
- Python 3.x
- Libraries: pandas, numpy

## Installation
1. Clone the repository.
 
    ```
2. Install the required libraries:
    ```bash
    pip install pandas numpy
    ```

3. Run in Jupyter notebook or Python Environment with installed modules. 
