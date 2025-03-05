import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collection import DataCollector
from signal_generation import SignalGenerator
from portfolio import PortfolioManager
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import warnings
import itertools
warnings.filterwarnings('ignore')

class Backtester:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.data_collector = DataCollector()
        self.signal_generator = SignalGenerator()
        self.portfolio_manager = PortfolioManager(initial_capital)
        self.performance_metrics = {}
        
    def run_backtest(self, start_date=None, end_date=None):
        """
        Run backtest over the specified period
        """
        # Fetch data
        data = self.data_collector.fetch_data(start_date, end_date)
        processed_data = self.data_collector.prepare_data(data)
        
        # Get benchmark data
        benchmark_data = self.data_collector.get_benchmark_data(start_date, end_date)
        
        # Initialize results tracking
        portfolio_values = []
        benchmark_values = []
        dates = []
        
        # Get all trading dates
        all_dates = sorted(list(next(iter(processed_data.values()))['historical'].index))
        
        # Monthly rebalancing
        current_month = None
        current_positions = {}
        
        # Track monthly performance
        monthly_returns = []
        monthly_dates = []
        
        for date in all_dates:
            # Check if we need to rebalance (monthly)
            if current_month is None or date.month != current_month:
                # Generate signals
                signals = self.signal_generator.generate_signals(processed_data)
                
                # Construct portfolio
                current_positions = self.portfolio_manager.construct_portfolio(signals, processed_data)
                current_month = date.month
                
                # Record monthly performance
                if len(portfolio_values) > 0:
                    monthly_return = (portfolio_values[-1] / portfolio_values[0] - 1) if len(monthly_returns) == 0 else \
                                   (portfolio_values[-1] / portfolio_values[-2] - 1)
                    monthly_returns.append(monthly_return)
                    monthly_dates.append(date)
                
            # Get current prices
            current_prices = {
                ticker: processed_data[ticker]['historical'].loc[date, 'Close']
                for ticker in processed_data.keys()
            }
            
            # Update positions (check stop losses)
            self.portfolio_manager.update_positions(current_prices)
            
            # Calculate portfolio value
            portfolio_value = self.portfolio_manager.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            
            # Calculate benchmark value
            if benchmark_data is not None and date in benchmark_data.index:
                benchmark_value = benchmark_data.loc[date, 'Close']
                benchmark_values.append(benchmark_value)
            else:
                benchmark_values.append(None)
                
            dates.append(date)
            
        # Calculate performance metrics
        self.calculate_performance_metrics(portfolio_values, benchmark_values, dates)
        
        return {
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'dates': dates,
            'metrics': self.performance_metrics,
            'trade_history': self.portfolio_manager.trade_history,
            'monthly_returns': monthly_returns,
            'monthly_dates': monthly_dates
        }
    
    def calculate_performance_metrics(self, portfolio_values, benchmark_values, dates):
        """
        Calculate key performance metrics
        """
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values)
        benchmark_values = np.array(benchmark_values)
        
        # Calculate returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        # Calculate cumulative returns
        portfolio_cumulative = (portfolio_values[-1] / self.initial_capital) - 1
        benchmark_cumulative = (benchmark_values[-1] / benchmark_values[0]) - 1
        
        # Calculate annualized returns
        years = (dates[-1] - dates[0]).days / 365
        portfolio_annualized = (1 + portfolio_cumulative) ** (1/years) - 1
        benchmark_annualized = (1 + benchmark_cumulative) ** (1/years) - 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = portfolio_returns - (risk_free_rate/252)  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Calculate alpha and beta
        excess_portfolio_returns = portfolio_returns - (risk_free_rate/252)
        excess_benchmark_returns = benchmark_returns - (risk_free_rate/252)
        
        beta = np.cov(excess_portfolio_returns, excess_benchmark_returns)[0,1] / np.var(excess_benchmark_returns)
        alpha = np.mean(excess_portfolio_returns) - beta * np.mean(excess_benchmark_returns)
        
        # Calculate additional metrics
        win_rate = len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns)
        profit_factor = abs(sum(r for r in portfolio_returns if r > 0) / sum(r for r in portfolio_returns if r < 0))
        
        # Store metrics
        self.performance_metrics = {
            'total_return': portfolio_cumulative,
            'annualized_return': portfolio_annualized,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'benchmark_return': benchmark_cumulative,
            'benchmark_annualized': benchmark_annualized,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
    def get_performance_summary(self):
        """
        Get a summary of the backtest performance
        """
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_manager.current_capital,
            'total_trades': len(self.portfolio_manager.trade_history),
            'metrics': self.performance_metrics
        }

class BacktestEngine:
    def __init__(self, signal_generator, portfolio_manager, initial_capital=1000000, transaction_cost=0.001):
        self.signal_generator = signal_generator
        self.portfolio_manager = portfolio_manager
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def prepare_walk_forward_periods(self, data, train_years, test_years):
        """
        Prepare walk-forward periods for analysis
        """
        # Get start and end dates
        start_date = data[list(data.keys())[0]].index[0]
        end_date = data[list(data.keys())[0]].index[-1]
        
        # Calculate total days and number of splits
        total_days = (end_date - start_date).days
        n_splits = max(2, int((total_days/365 - train_years)/test_years))
        
        # Create TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Convert dates to indices for splitting
        all_dates = data[list(data.keys())[0]].index
        date_indices = np.arange(len(all_dates))
        
        periods = []
        for train_idx, test_idx in tscv.split(date_indices):
            # Calculate train start and end dates
            train_start = all_dates[train_idx[0]]
            train_end = all_dates[train_idx[-1]]
            
            # Calculate test start and end dates
            test_start = all_dates[test_idx[0]]
            test_end = all_dates[test_idx[-1]]
            
            periods.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })
            
        return periods
        
    def backtest_period(self, data, period, is_training=False):
        """
        Run backtest for a single period
        """
        start_date, end_date = period
        
        # Initialize portfolio
        self.portfolio_manager.reset(self.initial_capital)
        
        # Initialize metrics tracking
        equity_curve = []
        drawdown = []
        trades = []
        current_high = self.initial_capital
        
        # Get dates for the period
        dates = data[list(data.keys())[0]].loc[start_date:end_date].index
        
        for date in dates:
            # Get current data
            current_data = {ticker: df.loc[:date] for ticker, df in data.items()}
            
            # Generate signals
            signals = self.signal_generator.generate_signals(current_data)
            
            # Update portfolio
            self.portfolio_manager.update_positions(current_data, date)
            
            # Construct portfolio if needed
            if not is_training:
                self.portfolio_manager.construct_portfolio(signals, current_data, date)
            
            # Track metrics
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            equity_curve.append(portfolio_value)
            
            # Calculate drawdown
            current_high = max(current_high, portfolio_value)
            current_drawdown = (current_high - portfolio_value) / current_high
            drawdown.append(current_drawdown)
            
            # Track trades
            if not is_training:
                new_trades = self.portfolio_manager.get_trades_for_date(date)
                trades.extend(new_trades)
        
        # Calculate performance metrics
        equity_curve = pd.Series(equity_curve, index=dates)
        drawdown = pd.Series(drawdown, index=dates)
        
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'equity_curve': equity_curve,
            'drawdown': drawdown,
            'total_return': (equity_curve[-1] / self.initial_capital) - 1,
            'annualized_return': ((equity_curve[-1] / self.initial_capital) ** (252/len(returns)) - 1),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            'max_drawdown': max(drawdown),
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
            'profit_factor': abs(sum(t['pnl'] for t in trades if t['pnl'] > 0)) / 
                           abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if trades else 0
        }
        
        return metrics
    
    def run_walk_forward_analysis(self, data, train_years, test_years):
        """
        Run walk-forward analysis
        """
        # Prepare periods for walk-forward analysis
        periods = self.prepare_walk_forward_periods(data, train_years, test_years)
        
        results = []
        for period in periods:
            # Run training period
            self.backtest_period(data, period['train'], is_training=True)
            
            # Run test period
            test_results = self.backtest_period(data, period['test'], is_training=False)
            results.append(test_results)
        
        # Combine results
        combined_results = {
            'equity_curve': pd.concat([r['equity_curve'] for r in results]),
            'drawdown': pd.concat([r['drawdown'] for r in results]),
            'total_return': results[-1]['total_return'],
            'annualized_return': results[-1]['annualized_return'],
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
            'max_drawdown': max(r['max_drawdown'] for r in results),
            'win_rate': np.mean([r['win_rate'] for r in results]),
            'profit_factor': np.mean([r['profit_factor'] for r in results])
        }
        
        return combined_results 