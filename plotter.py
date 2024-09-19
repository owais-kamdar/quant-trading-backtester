import matplotlib.pyplot as plt
import pandas as pd

def plot_performance(data, history, buy_signals, sell_signals, future_prices=None):
    plt.figure(figsize=(14, 7))

    # Plot stock price and portfolio value
    plt.plot(data.index, history, label='Portfolio Value')
    plt.plot(data.index, data['Close'], label='Stock Price', alpha=0.5)

    # Plot indicators: RSI and Bollinger Bands
    plt.plot(data.index, data['Upper_BB'], label='Upper Bollinger Band', linestyle='--', alpha=0.3)
    plt.plot(data.index, data['Lower_BB'], label='Lower Bollinger Band', linestyle='--', alpha=0.3)
    plt.plot(data.index, data['RSI'], label='RSI', color='purple', alpha=0.6)
    
    # Plot buy and sell signals
    for buy in buy_signals:
        plt.scatter(buy[0], buy[1], color='green', marker='^', label='Buy Signal')

    for sell in sell_signals:
        plt.scatter(sell[0], sell[1], color='red', marker='v', label='Sell Signal')

    # Plot future prices (optional)
    if future_prices is not None:
        future_dates = pd.date_range(start=data.index[-1], periods=len(future_prices) + 1, closed='right')
        plt.plot(future_dates, future_prices, label='Predicted Prices', color='orange')

    plt.title('Backtest Performance with Future Predictions')
    plt.legend()
    plt.show()
