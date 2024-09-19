


def backtest_strategy(data, initial_balance=10000, transaction_cost=0.001, stop_loss=0.05):
    """
    Simulates trading based on the Moving Average Crossover and incorporates additional indicators like RSI, Bollinger Bands, and MACD.
    :param data: Stock data with calculated indicators.
    :param initial_balance: Starting capital.
    :param transaction_cost: Transaction fee as a percentage of trade size.
    :param stop_loss: Max loss threshold.
    :return: Final balance, total return, history, and signals.
    """
    balance, shares, buy_signals, sell_signals, history = initial_balance, 0, [], [], []
    
    for i in range(len(data)):
        # Buy condition: MA crossover and RSI < 70 and Price below Bollinger Band upper bound
        if (data['50_MA'].iloc[i] > data['200_MA'].iloc[i]) and (data['RSI'].iloc[i] < 70) and (data['Close'].iloc[i] < data['Upper_BB'].iloc[i]) and shares == 0:
            shares = balance // data['Close'].iloc[i]
            balance -= shares * data['Close'].iloc[i] * (1 + transaction_cost)
            buy_signals.append((data.index[i], data['Close'].iloc[i]))

        # Sell condition: MA crossover and RSI > 30 and Price above Bollinger Band lower bound
        elif (data['50_MA'].iloc[i] < data['200_MA'].iloc[i]) and (data['RSI'].iloc[i] > 30) and (data['Close'].iloc[i] > data['Lower_BB'].iloc[i]) and shares > 0:
            balance += shares * data['Close'].iloc[i] * (1 - transaction_cost)
            shares = 0
            sell_signals.append((data.index[i], data['Close'].iloc[i]))

        # Stop loss implementation
        if shares > 0 and data['Close'].iloc[i] < buy_signals[-1][1] * (1 - stop_loss):
            balance += shares * data['Close'].iloc[i] * (1 - transaction_cost)
            shares = 0
            sell_signals.append((data.index[i], data['Close'].iloc[i]))

        # Log portfolio value
        portfolio_value = balance + shares * data['Close'].iloc[i]
        history.append(portfolio_value)

    return balance, history, buy_signals, sell_signals
