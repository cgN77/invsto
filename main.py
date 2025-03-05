from datetime import datetime, timedelta
from backtesting import Backtester
from performance import PerformanceAnalyzer

def main():
    # Set up backtest parameters
    initial_capital = 1000000
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print("Starting Alpha-Based Long-Short Strategy Backtest")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    
    # Run backtest
    backtester = Backtester(initial_capital)
    results = backtester.run_backtest(start_date, end_date)
    
    # Analyze results
    analyzer = PerformanceAnalyzer(results)
    
    # Generate and save plots
    print("\nGenerating performance plots...")
    analyzer.save_plots()
    
    # Print performance summary
    summary = backtester.get_performance_summary()
    print("\nPerformance Summary:")
    print("-" * 50)
    print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"Final Capital: ${summary['final_capital']:,.2f}")
    print(f"Total Return: {summary['metrics']['total_return']:.2%}")
    print(f"Annualized Return: {summary['metrics']['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {summary['metrics']['max_drawdown']:.2%}")
    print(f"Alpha: {summary['metrics']['alpha']:.2%}")
    print(f"Beta: {summary['metrics']['beta']:.2f}")
    print(f"Total Trades: {summary['total_trades']}")
    print("-" * 50)
    
    print("\nResults have been saved to the 'plots' directory.")
    print("Check the following files:")
    print("- portfolio_performance.png")
    print("- drawdown.png")
    print("- monthly_returns.png")
    print("- trade_distribution.png")
    print("- performance_report.csv")

if __name__ == "__main__":
    main() 