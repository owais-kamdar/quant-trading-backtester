# main.py

from strategy_optimizer import optimize_strategy
from data.data_fetcher import get_data
from backtester.strategy import backtest_strategy
from backtester.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr
from predictor.predictor import predict_future_prices
from plotter import plot_performance
import numpy as np
import ta

def main():
    try:
        # User input
        stock_symbols_input = input("Enter stock ticker symbols separated by commas (e.g., AAPL, MSFT): ")
        stock_symbols = [symbol.strip().upper() for symbol in stock_symbols_input.split(',')]
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        initial_balance = float(input("Enter initial balance: "))

        # Fetch and prepare data
        data_dict = get_data(stock_symbols, start=start_date, end=end_date)

        total_final_balance = 0
        total_initial_balance = initial_balance * len(stock_symbols)

        for symbol in stock_symbols:
            print(f"\nProcessing stock: {symbol}")
            data = data_dict[symbol]
            # Convert Spark DataFrame back to pandas DataFrame for backtesting
            data_pd = data.toPandas()
            data_pd.set_index('Date', inplace=True)

            # Optimize strategy parameters
            best_params, best_results = optimize_strategy(data_pd, initial_balance)
            final_balance, total_return, history, buy_signals, sell_signals, accuracy = best_results

            total_final_balance += final_balance

            # Performance metrics
            daily_returns = [0] + list(np.diff(history) / history[:-1])
            sharpe_ratio = calculate_sharpe_ratio(daily_returns)
            max_drawdown = calculate_max_drawdown(history)
            cagr = calculate_cagr(initial_balance, final_balance, len(data_pd) / 252)

            # Display results
            print(f"\nBest Parameters Found for {symbol}: {best_params}")
            print(f"\n=== Strategy Performance for {symbol} ===")
            print(f"Initial Investment: ${initial_balance:,.2f}")
            print(f"Final Portfolio Value: ${final_balance:,.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f} (Higher is better; >1 is considered good)")
            print(f"Max Drawdown: {max_drawdown:.2%} (Maximum loss from a peak to a trough)")
            print(f"CAGR: {cagr:.2%} (Annualized growth rate over the period)\n")
            print(f"Prediction Accuracy: {accuracy:.2f}% (Correct buy/sell predictions)\n")

            # Plot results
            plot_performance(data_pd, history, buy_signals, sell_signals)

            # Predict future prices using PyTorch
            print("\nPredicting future prices using PyTorch LSTM...")
            future_prices, lstm_metrics = predict_future_prices(data_pd)
            print("\n=== LSTM Model Evaluation ===")
            print(f"Test Loss (MSE): {lstm_metrics['Test Loss']:.4f}")
            print(f"Test MAE: {lstm_metrics['MAE']:.4f}")
            print(f"Test RMSE: {lstm_metrics['RMSE']:.4f}")
            print("Predicted Future Prices:", future_prices.flatten())

            # Plot results with future predictions
            plot_performance(data_pd, history, buy_signals, sell_signals, future_prices, symbol)

        # Overall portfolio performance
        overall_return = (total_final_balance - total_initial_balance) / total_initial_balance * 100
        print(f"\n=== Overall Portfolio Performance ===")
        print(f"Initial Total Investment: ${total_initial_balance:,.2f}")
        print(f"Final Total Portfolio Value: ${total_final_balance:,.2f}")
        print(f"Overall Total Return: {overall_return:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()