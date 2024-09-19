import matplotlib.pyplot as plt
import pandas as pd

def plot_performance(data, history, buy_signals, sell_signals, future_prices=None):
    plt.figure(figsize=(14, 7))

    # Plot stock price and portfolio value
    plt.plot(data.index, history, label='Portfolio Value')
    plt.plot(data.index, data['Close'], label='Stock Price', alpha=0.5)

    # Plot one "Buy" and one "Sell" signal only in the legend
    if buy_signals:
        plt.scatter(buy_signals[0][0], buy_signals[0][1], color='green', marker='^', label='Buy Signal')
    if sell_signals:
        plt.scatter(sell_signals[0][0], sell_signals[0][1], color='red', marker='v', label='Sell Signal')

    # Plot all the buy and sell signals (without adding more legend items)
    for buy in buy_signals:
        plt.scatter(buy[0], buy[1], color='green', marker='^')

    for sell in sell_signals:
        plt.scatter(sell[0], sell[1], color='red', marker='v')

    # Plot future prices (optional)
    if future_prices is not None:
        future_dates = pd.date_range(start=data.index[-1], periods=len(future_prices) + 1, closed='right')
        plt.plot(future_dates, future_prices, label='Predicted Prices', color='orange')

    # Title and legend
    plt.title('Backtest Performance with Future Predictions')
    plt.legend()

    # Display the plot
    plt.show()
