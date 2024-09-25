# data_fetcher.py

from pyspark.sql import SparkSession
import yfinance as yf
import pandas as pd
import ta  # Using 'ta' library as an alternative to TA-Lib

def get_data(stock_symbols, start, end):
    """
    Downloads historical stock data for multiple stocks and adds technical indicators.
    :param stock_symbols: List of ticker symbols of the stocks (e.g., ['AAPL', 'MSFT']).
    :param start: The start date for the data (YYYY-MM-DD).
    :param end: The end date for the data (YYYY-MM-DD).
    :return: A dictionary of Spark DataFrames containing stock data with indicators.
    """
    print(f"\nFetching data for {', '.join(stock_symbols)} from {start} to {end}...")

    # Initialize SparkSession
    spark = SparkSession.builder.appName("StockDataProcessing").getOrCreate()

    data_dict = {}
    for symbol in stock_symbols:
        # Download data using yfinance
        data = yf.download(symbol, start=start, end=end)
        data.reset_index(inplace=True)

        # Use 'ta' library to calculate indicators
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['100_MA'] = data['Close'].rolling(window=100).mean()
        data['200_MA'] = data['Close'].rolling(window=200).mean()
        data['EMA_50'] = ta.trend.EMAIndicator(close=data['Close'], window=50).ema_indicator()
        data['EMA_200'] = ta.trend.EMAIndicator(close=data['Close'], window=200).ema_indicator()
        data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
        bollinger = ta.volatility.BollingerBands(close=data['Close'], window=20)
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Middle'] = bollinger.bollinger_mavg()
        data['BB_Lower'] = bollinger.bollinger_lband()
        macd = ta.trend.MACD(close=data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()

        # Convert to Spark DataFrame
        sdf = spark.createDataFrame(data)
        data_dict[symbol] = sdf
        print(f"Data for {symbol} downloaded and processed.")

    print("All data downloaded and processed.\n")
    return data_dict
