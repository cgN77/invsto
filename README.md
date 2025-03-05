# Quantitative Trading Strategy

A sophisticated long-short equity trading strategy that combines multiple factors, dynamic risk management, and systematic hedging.

## Features

### Alpha Signal Generation

- Momentum factors (1-month and 3-month price changes)
- Mean reversion signals (RSI, Moving Average ratios)
- Volatility factors (standard deviation and daily range)
- Volume-based indicators (trends and volatility)

### Portfolio Construction

- Signal-based stock ranking and selection
- Dynamic position sizing based on signal strength
- Factor exposure neutralization
- Volatility scaling

### Risk Management

- ATR-based dynamic stop losses and take profits
- Volatility regime-based risk adjustment
- Maximum drawdown controls
- Position size limits
- Portfolio-level exposure constraints

### Execution & Hedging

- Correlation-based hedging with index futures
- Trailing stop management
- Systematic rebalancing
- Transaction cost consideration

## Components

- `portfolio.py`: Main portfolio management class
- `backtesting.py`: Backtesting engine with walk-forward analysis
- `signal_generation.py`: Alpha signal calculation
- `run_backtest.py`: Script to execute backtests

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository

```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the backtesting framework:

```bash
python run_backtest.py
```

## Configuration

Key parameters can be adjusted in the respective files:

- Position limits: 15% maximum per position
- Leverage: No leverage (1.0 maximum)
- Maximum drawdown: 15%
- Stop loss: 2 \* ATR
- Take profit: 4 \* ATR
- Trailing stop: Activates at 2% profit
- Minimum profit lock: 1%
- Correlation threshold for hedging: 0.7
- Maximum hedge ratio: 50%

## License

MIT License
