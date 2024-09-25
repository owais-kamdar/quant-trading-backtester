# metrics.py

import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculates the Sharpe ratio to measure risk-adjusted return.
    :param returns: Daily returns of the portfolio.
    :param risk_free_rate: The risk-free rate of return (e.g., treasury bonds).
    :return: The Sharpe ratio.
    """
    excess_returns = np.array(returns) - risk_free_rate
    std_dev = np.std(excess_returns)
    return np.mean(excess_returns) / std_dev if std_dev != 0 else 0

def calculate_max_drawdown(history):
    """
    Calculates the maximum drawdown (biggest peak-to-trough drop).
    :param history: The portfolio value over time.
    :return: The maximum drawdown.
    """
    peak = history[0]
    max_drawdown = 0
    for value in history:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_cagr(initial_value, final_value, periods):
    """
    Calculates the CAGR of the portfolio.
    :param initial_value: Starting value of the portfolio.
    :param final_value: Ending value of the portfolio.
    :param periods: Total time periods (in years).
    :return: The CAGR, or 0 if periods or initial_value are invalid.
    """
    if initial_value <= 0 or periods <= 0:
        return 0  # Avoid division by zero or invalid CAGR
    return (final_value / initial_value) ** (1 / periods) - 1
