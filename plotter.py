# plotter.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_performance(data, history, buy_signals, sell_signals, future_prices=None, symbol='Stock'):
    plt.figure(figsize=(14, 7))

    # Plot stock price and portfolio value
    plt.plot(data.index, history, label='Portfolio Value')
    plt.plot(data.index, data['Close'], label='Stock Price', alpha=0.5)

    # Plot buy and sell signals
    for buy in buy_signals:
        plt.scatter(buy[0], buy[1], color='green', marker='^', label='Buy Signal')
    for sell in sell_signals:
        plt.scatter(sell[0], sell[1], color='red', marker='v', label='Sell Signal')

    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Plot future prices (optional)
    if future_prices is not None:
        future_dates = pd.date_range(start=data.index[-1], periods=len(future_prices) + 1, closed='right')
        plt.plot(future_dates, future_prices, label='Predicted Prices', color='orange')
        plt.axvline(x=data.index[-1], color='grey', linestyle='--')  # Line separating historical and predicted

    # Title and labels
    plt.title(f'{symbol} Backtest Performance with LSTM Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price / Portfolio Value')
    plt.legend()

    # Display the plot
    plt.show()
