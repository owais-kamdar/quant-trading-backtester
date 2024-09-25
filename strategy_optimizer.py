# strategy_optimizer.py

import itertools
from backtester.metrics import calculate_sharpe_ratio
from backtester.strategy import backtest_strategy
import numpy as np
import ta

def optimize_strategy(data, initial_balance):
    # Parameter ranges
    short_ma_periods = [20, 50, 100]
    long_ma_periods = [100, 200, 250]
    rsi_thresholds = [(30, 70), (20, 80)]
    stop_losses = [0.02, 0.05]
    take_profits = [0.05, 0.10, 0.15]

    best_sharpe = float('-inf')
    best_params = None
    best_results = None

    # Iterate over all combinations
    for short_ma_period, long_ma_period, (rsi_lower, rsi_upper), stop_loss, take_profit in itertools.product(
        short_ma_periods, long_ma_periods, rsi_thresholds, stop_losses, take_profits):

        # Skip invalid MA combinations
        if short_ma_period >= long_ma_period:
            continue

        # Prepare data with new MAs and RSI
        data['Short_MA'] = data['Close'].rolling(window=short_ma_period).mean()
        data['Long_MA'] = data['Close'].rolling(window=long_ma_period).mean()
        data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()

        # Backtest with new parameters
        results = backtest_strategy(
            data,
            initial_balance=initial_balance,
            transaction_cost=0.001,
            stop_loss=stop_loss,
            take_profit=take_profit,
            short_ma='Short_MA',
            long_ma='Long_MA',
            rsi_lower=rsi_lower,
            rsi_upper=rsi_upper
        )

        final_balance, total_return, history, _, _, _ = results
        daily_returns = [0] + list(np.diff(history) / history[:-1])
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)

        # Check if this is the best Sharpe Ratio so far
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_params = {
                'short_ma_period': short_ma_period,
                'long_ma_period': long_ma_period,
                'rsi_lower': rsi_lower,
                'rsi_upper': rsi_upper,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            best_results = results

    return best_params, best_results
