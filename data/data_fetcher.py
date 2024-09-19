import yfinance as yf
import talib
import pandas as pd

import yfinance as yf
import talib
import pandas as pd

def get_data(stock_symbol, start, end):
    """
    Downloads historical stock data and adds technical indicators like RSI, Bollinger Bands, EMA, and MACD.
    :param stock_symbol: The ticker symbol of the stock (e.g., 'AAPL').
    :param start: The start date for the data (YYYY-MM-DD).
    :param end: The end date for the data (YYYY-MM-DD).
    :return: A pandas DataFrame containing stock data with indicators.
    """
    print(f"\nFetching data for {stock_symbol} from {start} to {end}...")
    data = yf.download(stock_symbol, start=start, end=end)

    # Calculate Moving Averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['100_MA'] = data['Close'].rolling(window=100).mean()  # Added 100-day moving average
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    # Calculate Exponential Moving Average (EMA)
    data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)
    data['EMA_200'] = talib.EMA(data['Close'], timeperiod=200)

    # Calculate RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

    # Calculate Bollinger Bands
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Calculate MACD (Moving Average Convergence Divergence)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    print("Data download complete with technical indicators.\n")
    return data
