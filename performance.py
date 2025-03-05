import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, backtest_results):
        self.backtest_results = backtest_results
        self.portfolio_values = backtest_results['portfolio_values']
        self.benchmark_values = backtest_results['benchmark_values']
        self.dates = backtest_results['dates']
        self.metrics = backtest_results['metrics']
        self.trade_history = backtest_results['trade_history']
        
    def plot_portfolio_performance(self):
        """
        Plot portfolio and benchmark performance over time
        """
        plt.figure(figsize=(12, 6))
        
        # Normalize values to 100 at start
        portfolio_normalized = np.array(self.portfolio_values) / self.portfolio_values[0] * 100
        benchmark_normalized = np.array(self.benchmark_values) / self.benchmark_values[0] * 100
        
        plt.plot(self.dates, portfolio_normalized, label='Portfolio', linewidth=2)
        plt.plot(self.dates, benchmark_normalized, label='S&P 500', linewidth=2)
        
        plt.title('Portfolio Performance vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value (100 = Start)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        return plt.gcf()
    
    def plot_drawdown(self):
        """
        Plot portfolio drawdown over time
        """
        portfolio_values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, drawdowns, label='Drawdown', linewidth=2)
        
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        return plt.gcf()
    
    def plot_monthly_returns(self):
        """
        Plot monthly returns heatmap
        """
        # Convert dates to datetime if they aren't already
        dates = pd.to_datetime(self.dates)
        
        # Calculate monthly returns
        portfolio_returns = pd.Series(self.portfolio_values, index=dates).pct_change()
        monthly_returns = portfolio_returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).mean()
        
        # Reshape the data for the heatmap
        pivot_table = pivot_table.unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        return plt.gcf()
    
    def plot_trade_distribution(self):
        """
        Plot distribution of trade returns
        """
        trade_returns = [trade['pnl'] / (trade['entry_price'] * trade['shares']) for trade in self.trade_history]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(trade_returns, bins=50)
        plt.title('Distribution of Trade Returns')
        plt.xlabel('Return')
        plt.ylabel('Count')
        
        return plt.gcf()
    
    def generate_performance_report(self):
        """
        Generate a comprehensive performance report
        """
        trade_returns = [trade['pnl'] / (trade['entry_price'] * trade['shares']) for trade in self.trade_history]
        
        report = {
            'Performance Metrics': {
                'Total Return': f"{self.metrics['total_return']:.2%}",
                'Annualized Return': f"{self.metrics['annualized_return']:.2%}",
                'Sharpe Ratio': f"{self.metrics['sharpe_ratio']:.2f}",
                'Maximum Drawdown': f"{self.metrics['max_drawdown']:.2%}",
                'Alpha': f"{self.metrics['alpha']:.2%}",
                'Beta': f"{self.metrics['beta']:.2f}",
                'Benchmark Return': f"{self.metrics['benchmark_return']:.2%}",
                'Benchmark Annualized': f"{self.metrics['benchmark_annualized']:.2%}"
            },
            'Trading Statistics': {
                'Total Trades': len(self.trade_history),
                'Average Trade Return': f"{np.mean(trade_returns):.2%}",
                'Win Rate': f"{len([t for t in self.trade_history if t['pnl'] > 0]) / len(self.trade_history):.2%}",
                'Profit Factor': f"{abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0) / sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0)):.2f}"
            }
        }
        
        return report
    
    def save_plots(self, directory='plots'):
        """
        Save all plots to a directory
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.plot_portfolio_performance().savefig(f'{directory}/portfolio_performance.png')
        self.plot_drawdown().savefig(f'{directory}/drawdown.png')
        self.plot_monthly_returns().savefig(f'{directory}/monthly_returns.png')
        self.plot_trade_distribution().savefig(f'{directory}/trade_distribution.png')
        
        # Save performance report to CSV
        report = self.generate_performance_report()
        pd.DataFrame(report).to_csv(f'{directory}/performance_report.csv') 