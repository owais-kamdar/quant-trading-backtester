from data.data_loader import get_data
from strategy.backtester import backtest_strategy
from strategy.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr
from prediction.stock_prediction import predict_future_prices
from visualization.plotter import plot_performance

def main():
    # User input
    stock_symbol = input("Enter stock ticker (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    initial_balance = float(input("Enter initial balance: "))

    # Fetch and prepare data
    data = get_data(stock_symbol, start=start_date, end=end_date)
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    # Backtest strategy
    final_balance, history, buy_signals, sell_signals = backtest_strategy(data, initial_balance)

    # Performance metrics
    daily_returns = np.diff(history) / history[:-1]
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(history)
    cagr = calculate_cagr(initial_balance, final_balance, len(data) / 252)

    # Display results
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"CAGR: {cagr:.2%}")

    # Plot results
    plot_performance(data, history, buy_signals, sell_signals)

    # Predict future prices using LSTM
    print("\nPredicting future prices using LSTM...")
    future_prices = predict_future_prices(data)
    print("Predicted Future Prices:", future_prices)

if __name__ == '__main__':
    main()
