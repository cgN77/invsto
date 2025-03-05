import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from signal_generation import SignalGenerator
from portfolio import PortfolioManager
from backtesting import BacktestEngine

def load_data():
    """
    Load and prepare historical price data
    """
    # For demo, create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    data = {}
    for stock in stocks:
        # Generate random price series with trend and volatility
        price = 100
        prices = []
        for i in range(len(dates)):
            # Add random walk with drift
            drift = 0.0002  # Small positive drift
            volatility = 0.02  # Daily volatility
            price *= np.exp(drift + volatility * np.random.normal())
            prices.append(price)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Volume': [abs(np.random.normal(1000000, 200000)) for _ in prices]
        }, index=dates)
        
        data[stock] = df
    
    return data

def plot_results(results):
    """
    Plot backtest results
    """
    plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results['equity_curve'], label='Portfolio Value')
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    plt.fill_between(results['drawdown'].index, results['drawdown'].values, 0, 
                     color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown %')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()

def print_performance_metrics(results):
    """
    Print key performance metrics
    """
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print("-" * 50)

def main():
    # Load historical data
    print("Loading historical data...")
    data = load_data()
    
    # Initialize components
    signal_generator = SignalGenerator()
    portfolio_manager = PortfolioManager()
    backtest_engine = BacktestEngine(
        signal_generator=signal_generator,
        portfolio_manager=portfolio_manager,
        initial_capital=1000000,
        transaction_cost=0.001
    )
    
    # Set up walk-forward analysis parameters
    train_years = 2
    test_years = 1
    
    # Run walk-forward analysis
    print("Running walk-forward analysis...")
    results = backtest_engine.run_walk_forward_analysis(
        data=data,
        train_years=train_years,
        test_years=test_years
    )
    
    # Plot and print results
    plot_results(results)
    print_performance_metrics(results)
    
    # Save results
    results['equity_curve'].to_csv('equity_curve.csv')
    
    print("\nBacktest completed successfully!")
    print("Results have been saved to 'backtest_results.png' and 'equity_curve.csv'")

if __name__ == "__main__":
    main() 