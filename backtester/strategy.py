# strategy.py

def backtest_strategy(
    data, initial_balance=10000, transaction_cost=0.001,
    stop_loss=0.05, take_profit=0.15, leverage=1,
    short_ma='Short_MA', long_ma='Long_MA', rsi_lower=20, rsi_upper=80):
    print("Backtesting the strategy...\n")
    balance = initial_balance
    shares = 0
    buy_price = 0
    history = []
    buy_signals = []
    sell_signals = []
    correct_predictions = 0
    total_predictions = 0
    max_portfolio_value = initial_balance

    for i in range(len(data)):
        # Enhanced Buy signal
        if (data[short_ma].iloc[i] > data[long_ma].iloc[i]
            and shares == 0
            and data['RSI'].iloc[i] < rsi_lower):

            shares = (balance // data['Close'].iloc[i]) * leverage
            balance -= shares * data['Close'].iloc[i] * (1 + transaction_cost)
            buy_price = data['Close'].iloc[i]
            buy_signals.append((data.index[i], data['Close'].iloc[i]))
            total_predictions += 1

            # Check if the stock price increased within 5 days after the buy
            if i + 5 < len(data) and data['Close'].iloc[i + 5] > data['Close'].iloc[i]:
                correct_predictions += 1

        # Enhanced Sell signal
        elif (data[short_ma].iloc[i] < data[long_ma].iloc[i]
              and shares > 0
              and data['RSI'].iloc[i] > rsi_upper):

            balance += shares * data['Close'].iloc[i] * (1 - transaction_cost)
            shares = 0
            sell_signals.append((data.index[i], data['Close'].iloc[i]))
            total_predictions += 1

            # Check if the stock price decreased within 5 days after the sell
            if i + 5 < len(data) and data['Close'].iloc[i + 5] < data['Close'].iloc[i]:
                correct_predictions += 1

        # Stop-loss: exit if the price drops below a certain percentage of buy price
        if shares > 0 and data['Close'].iloc[i] < buy_price * (1 - stop_loss):
            balance += shares * data['Close'].iloc[i] * (1 - transaction_cost)
            shares = 0
            sell_signals.append((data.index[i], data['Close'].iloc[i]))

        # Take-profit: exit if the price rises above a certain percentage of buy price
        if shares > 0 and data['Close'].iloc[i] > buy_price * (1 + take_profit):
            balance += shares * data['Close'].iloc[i] * (1 - transaction_cost)
            shares = 0
            sell_signals.append((data.index[i], data['Close'].iloc[i]))

        # Track maximum portfolio value for drawdown
        portfolio_value = balance + shares * data['Close'].iloc[i]
        history.append(portfolio_value)
        max_portfolio_value = max(max_portfolio_value, portfolio_value)

    # Calculate final balance, total return, and accuracy
    final_balance = balance + shares * data['Close'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return final_balance, total_return, history, buy_signals, sell_signals, accuracy
