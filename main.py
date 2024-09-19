from data.data_fetcher import get_data
from backtester.strategy import backtest_strategy
from backtester.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr
from predictor.predictor import predict_future_prices
from plotter import plot_performance
import numpy as np

def main():
    try:
        # User input
        stock_symbol = input("Enter stock ticker (e.g., AAPL): ")
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        initial_balance = float(input("Enter initial balance: "))

        # Fetch and prepare data
        data = get_data(stock_symbol, start=start_date, end=end_date)

        # Backtest strategy
        final_balance, total_return, history, buy_signals, sell_signals, accuracy = backtest_strategy(data, initial_balance)

        # Performance metrics
        daily_returns = [0] + list(np.diff(history) / history[:-1])
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        max_drawdown = calculate_max_drawdown(history)
        cagr = calculate_cagr(initial_balance, final_balance, len(data) / 252)

        # Display results
        print(f"\n=== Strategy Performance for {stock_symbol} ===")
        print(f"Initial Investment: ${initial_balance:,.2f}")
        print(f"Final Portfolio Value: ${final_balance:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f} (Higher is better; >1 is considered good)")
        print(f"Max Drawdown: {max_drawdown:.2%} (Maximum loss from a peak to a trough)")
        print(f"CAGR: {cagr:.2%} (Annualized growth rate over the period)\n")
        print(f"Prediction Accuracy: {accuracy:.2f}% (Correct buy/sell predictions)\n")

        # Plot results
        plot_performance(data, history, buy_signals, sell_signals)

        # Predict future prices using LSTM
        print("\nPredicting future prices using LSTM...")
        future_prices = predict_future_prices(data)
        print("Predicted Future Prices:", future_prices)

        plot_performance(data, history, buy_signals, sell_signals, future_prices)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
