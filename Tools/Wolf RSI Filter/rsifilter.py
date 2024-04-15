import yfinance as yf
import pandas as pd

nasdaq100_stocks = [
    'ADBE', 'AMD', 'ALGN', 'GOOGL', 'GOOG', 'AMZN', 'AMGN', 'ADI',
    'AAPL', 'AMAT', 'ASML', 'TEAM', 'ADSK', 'ADP', 'BIDU', 'BIIB', 'BMRN', 'BKNG',
    'AVGO', 'CDNS', 'CDW',  'CHTR', 'CHKP', 'CTAS', 'CSCO',  'CTSH',
    'CMCSA', 'CPRT', 'COST', 'CSX', 'DXCM', 'DOCU', 'DLTR', 'EBAY', 'EA', 'EXC',
    'META', 'FAST', 'FOXA', 'FOX', 'GILD', 'IDXX', 'ILMN', 'INCY', 'INTC',
    'INTU', 'ISRG', 'JD', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LULU', 'MAR', 'MRVL', 'MTCH',
    'MELI', 'MCHP', 'MU', 'MSFT', 'MRNA', 'MDLZ', 'MNST', 'NTES', 'NFLX',
    'NVDA', 'NXPI', 'ORLY', 'OKTA', 'PCAR', 'PAYX', 'PYPL', 'PTON', 'PEP', 'PDD',
     'QCOM', 'REGN', 'ROST',  'SIRI', 'SWKS', 'SPLK', 'SBUX', 'SNPS',
    'TMUS', 'TSLA', 'TXN', 'TCOM', 'VRSN', 'VRSK', 'VRTX', 'WBA', 'WDAY', 'XEL',
    'ZM', 'META'
]

def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_stocks_below_rsi(symbol_list, rsi_threshold=30, history_days=30):
    result = []
    for symbol in symbol_list:
        try:
            stock_data = yf.download(symbol, period='60d')
            if len(stock_data) < history_days:
                continue
            stock_data['RSI'] = calculate_rsi(stock_data)
            if stock_data['RSI'].iloc[-1] <= rsi_threshold:
                result.append(symbol)
        except:
            continue
    return result

# Example usage
stocks_below_rsi = find_stocks_below_rsi(nasdaq100_stocks)

print("Stocks below 30 RSI:")
for stock in stocks_below_rsi:
    print(stock)
