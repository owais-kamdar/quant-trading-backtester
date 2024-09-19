
# Backtesting Quantitative Trading Strategy

## Overview
This Python-based framework simulates and evaluates a **Moving Average Crossover trading strategy** using historical stock data. The framework integrates key performance metrics such as the **Sharpe Ratio**, **Max Drawdown**, and **Compound Annual Growth Rate (CAGR)**, along with realistic risk management techniques like **stop-loss**, **take-profit**, and **transaction costs**. Dynamic data visualizations display buy/sell signals and portfolio performance over time, allowing users to assess the strategy’s effectiveness.

## Features
- **Backtest Strategy:** Simulates a Moving Average Crossover strategy with additional indicators like RSI and Bollinger Bands.
- **Performance Metrics:** Calculates Sharpe Ratio, Max Drawdown, and CAGR to evaluate risk-adjusted returns.
- **Risk Management:** Incorporates stop-loss, take-profit levels, and transaction costs to reflect real-world trading conditions.
- **Data Visualization:** Plots stock prices, portfolio value, and buy/sell signals, allowing for easy interpretation.
- **Prediction:** Predicts future stock prices using LSTM models.

## Prerequisites
To run this project, you need the following libraries installed:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `TA-Lib`
- `tensorflow`

You can install these libraries using `pip`:

```bash
pip install yfinance pandas numpy matplotlib TA-Lib tensorflow
```

## Setup
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/owais-kamdar/trading-backtest.git
   ```

2. Navigate to the project directory:

   ```bash
   cd trading-backtest
   ```

3. Install the required libraries (if not already installed):

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Python script:

   ```bash
   python main.py

   ```

## Usage
1. When prompted, enter the ticker symbol of the stock you'd like to backtest (e.g., AAPL).
2. Enter the start date and end date for the historical data (format: YYYY-MM-DD).
3. Provide the initial investment amount for the simulation.

The program will backtest the strategy on the provided data and display key metrics like Sharpe Ratio, Max Drawdown, and CAGR. Additionally, it will plot the portfolio value and buy/sell signals over time.

## Example
```bash
Enter the stock ticker symbol (e.g., AAPL): AAPL
Enter the start date (YYYY-MM-DD): 2000-01-01
Enter the end date (YYYY-MM-DD): 2020-01-01
Enter the initial investment amount ($): 10000
```

## Output
- Final Portfolio Value: $X,XXX.XX
- Total Return: XX.XX%
- Sharpe Ratio: X.XX
- Max Drawdown: X.XX%
- CAGR: X.XX%
- Buy/Sell signals plot

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
