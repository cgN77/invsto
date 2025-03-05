import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PortfolioManager:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.hedge_positions = {}  # Track index hedge positions
        self.trades = []
        self.stop_loss_pct = 0.02  # Stop loss percentage
        self.take_profit_pct = 0.05  # Take profit percentage
        self.max_position_size = 0.15  # Maximum position size as fraction of capital
        self.max_leverage = 1.0  # No leverage
        self.max_drawdown = 0.15  # Maximum 15% drawdown
        self.min_capital = initial_capital * 0.85  # Minimum capital threshold
        self.factor_exposure_threshold = 0.3  # Maximum allowed factor exposure
        
        # Correlation and hedging parameters
        self.correlation_threshold = 0.7  # Threshold for adding hedges
        self.correlation_window = 60  # Rolling window for correlation calculation
        self.max_hedge_ratio = 0.5  # Maximum hedge size as % of portfolio
        self.min_hedge_ratio = 0.1  # Minimum hedge size when activated
        self.hedge_adjustment_factor = 0.2  # How quickly to adjust hedge size
        
        # ATR-based risk parameters
        self.atr_period = 14  # Period for ATR calculation
        self.atr_stop_multiplier = 2.0  # Multiplier for ATR-based stops
        self.atr_take_profit_multiplier = 4.0  # Multiplier for ATR-based take profits
        self.trailing_stop_activation = 0.02  # Profit level to activate trailing stop
        self.min_profit_lock = 0.01  # Minimum profit to lock in (1%)
        
    def calculate_factor_exposures(self, data, tickers):
        """
        Calculate factor exposures using technical data
        """
        factor_data = {}
        
        for ticker in tickers:
            # Calculate various factors
            hist_data = data[ticker]
            
            # Calculate returns
            returns = hist_data['Close'].pct_change()
            
            factors = {
                # Momentum factors
                'momentum_1m': hist_data['Close'].pct_change(periods=20).iloc[-1],
                'momentum_3m': hist_data['Close'].pct_change(periods=60).iloc[-1],
                
                # Volatility factors
                'volatility': returns.std() * np.sqrt(252),
                'daily_range': ((hist_data['High'] - hist_data['Low']) / hist_data['Close']).mean(),
                
                # Volume factors
                'volume_trend': hist_data['Volume'].pct_change(periods=20).mean(),
                'volume_std': hist_data['Volume'].pct_change().std(),
                
                # Technical factors
                'ma_ratio': hist_data['Close'].iloc[-1] / hist_data['Close'].rolling(window=50).mean().iloc[-1],
                'rsi': self._calculate_rsi(hist_data['Close']),
                'price_trend': hist_data['Close'].pct_change(periods=10).mean()
            }
            
            factor_data[ticker] = factors
            
        # Convert to DataFrame and handle missing values
        factor_df = pd.DataFrame(factor_data).T
        factor_df = factor_df.fillna(factor_df.mean())
        
        # Standardize factors
        scaler = StandardScaler()
        standardized_factors = scaler.fit_transform(factor_df)
        
        # Run PCA to identify main risk factors
        pca = PCA(n_components=0.95)  # Capture 95% of variance
        pca_factors = pca.fit_transform(standardized_factors)
        
        return pd.DataFrame(pca_factors, index=factor_df.index), pca.components_, pca.explained_variance_ratio_
        
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
        
    def neutralize_factor_exposures(self, weights, factor_exposures, components):
        """
        Adjust weights to reduce factor exposures while maintaining signal alpha
        """
        weights_series = pd.Series(weights)
        initial_exposures = factor_exposures.T @ weights_series
        
        # Iteratively adjust weights to reduce factor exposures
        adjusted_weights = weights_series.copy()
        max_iterations = 50
        
        for _ in range(max_iterations):
            # Calculate current factor exposures
            current_exposures = factor_exposures.T @ adjusted_weights
            
            # Check if all exposures are within threshold
            if all(abs(exp) < self.factor_exposure_threshold for exp in current_exposures):
                break
                
            # Find the factor with highest absolute exposure
            worst_factor = abs(current_exposures).argmax()
            
            # Calculate adjustment needed
            exposure_adjustment = -current_exposures[worst_factor] * 0.5
            factor_weights = pd.Series(factor_exposures.iloc[:, worst_factor])
            
            # Adjust weights
            weight_adjustments = factor_weights * exposure_adjustment
            adjusted_weights += weight_adjustments
            
            # Ensure weights remain within bounds
            adjusted_weights = adjusted_weights.clip(-self.max_position_size, self.max_position_size)
            
            # Re-normalize weights
            adjusted_weights = adjusted_weights / abs(adjusted_weights).sum()
            
        return adjusted_weights.to_dict()
        
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def calculate_dynamic_risk_levels(self, data, direction='long'):
        """
        Calculate dynamic risk levels based on ATR and market conditions
        """
        # Calculate ATR
        atr = self.calculate_atr(data, self.atr_period)
        
        # Calculate volatility regime
        returns = data['Close'].pct_change()
        current_vol = returns.rolling(window=20).std() * np.sqrt(252)
        historical_vol = returns.rolling(window=252).std() * np.sqrt(252)
        vol_ratio = current_vol / historical_vol
        
        # Adjust ATR multipliers based on volatility regime
        if vol_ratio.iloc[-1] > 1.5:  # High volatility
            stop_multiplier = self.atr_stop_multiplier * 1.5
            take_profit_multiplier = self.atr_take_profit_multiplier * 1.3
        elif vol_ratio.iloc[-1] < 0.7:  # Low volatility
            stop_multiplier = self.atr_stop_multiplier * 0.8
            take_profit_multiplier = self.atr_take_profit_multiplier * 0.9
        else:  # Normal volatility
            stop_multiplier = self.atr_stop_multiplier
            take_profit_multiplier = self.atr_take_profit_multiplier
        
        current_atr = atr.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if direction == 'long':
            stop_loss = current_price - (stop_multiplier * current_atr)
            take_profit = current_price + (take_profit_multiplier * current_atr)
        else:  # short
            stop_loss = current_price + (stop_multiplier * current_atr)
            take_profit = current_price - (take_profit_multiplier * current_atr)
            
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': current_atr,
            'stop_multiplier': stop_multiplier,
            'take_profit_multiplier': take_profit_multiplier
        }
        
    def update_trailing_stop(self, position, current_price, atr):
        """
        Update trailing stop based on price movement and ATR
        """
        entry_price = position['entry_price']
        direction = position['direction']
        initial_stop = position['initial_stop']
        current_stop = position['stop_loss']
        
        # Calculate current profit/loss
        if direction == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            # Update trailing stop if profit exceeds activation threshold
            if profit_pct > self.trailing_stop_activation:
                # New stop is maximum of (current price - ATR) and current stop
                new_stop = max(
                    current_price - (self.atr_stop_multiplier * atr),
                    current_stop,
                    entry_price + (entry_price * self.min_profit_lock)  # Lock in minimum profit
                )
                return new_stop
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
            # Update trailing stop if profit exceeds activation threshold
            if profit_pct > self.trailing_stop_activation:
                # New stop is minimum of (current price + ATR) and current stop
                new_stop = min(
                    current_price + (self.atr_stop_multiplier * atr),
                    current_stop,
                    entry_price - (entry_price * self.min_profit_lock)  # Lock in minimum profit
                )
                return new_stop
                
        return current_stop
        
    def construct_portfolio(self, signals, data, current_date):
        """
        Construct portfolio based on signals with adaptive position sizing and factor neutrality
        """
        # Check if we've hit maximum drawdown
        if self.current_capital < self.min_capital:
            print(f"Maximum drawdown reached. Closing all positions.")
            print(f"Current capital: {self.current_capital:.2f}, Min capital: {self.min_capital:.2f}")
            print(f"Drawdown: {((self.initial_capital - self.current_capital) / self.initial_capital * 100):.2f}%")
            
            # Close all positions but don't reset yet
            self.close_all_positions(data, current_date)
            
            # Only reset if we're still below min_capital after closing positions
            if self.current_capital < self.min_capital:
                print("Still below minimum capital after closing positions. Resetting portfolio.")
                self.reset(self.initial_capital)
                return {}
            else:
                print("Recovered above minimum capital after closing positions. Continuing trading.")
            
        # Get composite scores
        composite_scores = signals['composite']
        
        # Sort stocks by absolute composite score
        sorted_stocks = sorted(composite_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Select top stocks for long/short based on sign of signal
        long_candidates = [(ticker, score) for ticker, score in sorted_stocks if score > 0][:3]
        short_candidates = [(ticker, score) for ticker, score in sorted_stocks if score < 0][:3]
        
        # If no valid signals, return empty portfolio
        if not long_candidates and not short_candidates:
            return {}
            
        # Calculate initial position weights using wi = Si/Σ|S| formula
        all_positions = long_candidates + short_candidates
        total_signal = sum(abs(score) for _, score in all_positions)
        
        # Calculate initial weights
        position_weights = {}
        for ticker, score in all_positions:
            # Base weight from signal strength
            weight = abs(score) / total_signal if total_signal > 0 else 1.0 / len(all_positions)
            position_weights[ticker] = weight * (1 if score > 0 else -1)  # Include direction in weight
            
        # Calculate factor exposures
        factor_exposures, components, variance_explained = self.calculate_factor_exposures(
            data, position_weights.keys()
        )
        
        # Neutralize factor exposures
        neutral_weights = self.neutralize_factor_exposures(
            position_weights, factor_exposures, components
        )
        
        # Apply volatility scaling to neutral weights
        final_weights = {}
        for ticker, weight in neutral_weights.items():
            # Calculate volatility scalar
            vol = data[ticker]['Close'].pct_change().std() * np.sqrt(252)
            target_vol = 0.20  # Target annualized volatility of 20%
            vol_scalar = min(target_vol / (vol + 1e-6), 2.0)  # Cap scaling at 2x
            
            # Apply scaling and position size cap
            final_weights[ticker] = min(
                abs(weight) * vol_scalar,
                self.max_position_size
            ) * np.sign(weight)
            
        # Normalize weights to ensure they sum to 1 while preserving direction
        total_long = sum(w for w in final_weights.values() if w > 0)
        total_short = abs(sum(w for w in final_weights.values() if w < 0))
        total = max(total_long, total_short) * 2
        
        final_weights = {t: w/total for t, w in final_weights.items()}
        
        # Calculate position sizes and create positions
        self.positions = {}
        available_capital = self.current_capital
        
        for ticker, weight in final_weights.items():
            direction = 'long' if weight > 0 else 'short'
            position_size = available_capital * abs(weight)
            
            # Calculate dynamic risk levels
            risk_levels = self.calculate_dynamic_risk_levels(
                data[ticker],
                direction
            )
            
            current_price = data[ticker]['Close'].iloc[-1]
            shares = position_size / current_price
            cost_basis = shares * current_price
            
            self.positions[ticker] = {
                'shares': shares,
                'direction': direction,
                'entry_price': current_price,
                'initial_stop': risk_levels['stop_loss'],
                'stop_loss': risk_levels['stop_loss'],
                'take_profit': risk_levels['take_profit'],
                'position_size': position_size,
                'weight': weight,
                'signal_score': composite_scores[ticker],
                'factor_exposures': factor_exposures.loc[ticker].to_dict(),
                'atr': risk_levels['atr'],
                'stop_multiplier': risk_levels['stop_multiplier'],
                'take_profit_multiplier': risk_levels['take_profit_multiplier'],
                'cost_basis': cost_basis,
                'unrealized_pnl': 0,  # Initialize with 0
                'entry_date': current_date
            }
            
        return self.positions
    
    def calculate_portfolio_correlations(self, data):
        """
        Calculate rolling correlations between portfolio positions and detect sector/market exposures
        """
        # Extract price series for all positions
        price_data = {}
        for ticker in self.positions:
            price_data[ticker] = data[ticker]['Close']
        
        if not price_data:
            return None, 0
            
        # Create price DataFrame and calculate returns
        prices_df = pd.DataFrame(price_data)
        returns_df = prices_df.pct_change()
        
        # Calculate rolling correlations between long and short positions
        long_positions = [t for t, p in self.positions.items() if p['direction'] == 'long']
        short_positions = [t for t, p in self.positions.items() if p['direction'] == 'short']
        
        if not long_positions or not short_positions:
            return None, 0
            
        long_returns = returns_df[long_positions].mean(axis=1)
        short_returns = returns_df[short_positions].mean(axis=1)
        
        rolling_corr = long_returns.rolling(window=self.correlation_window).corr(short_returns)
        
        # Calculate average pairwise correlation
        all_corr = returns_df.rolling(window=self.correlation_window).corr()
        avg_pairwise_corr = all_corr.groupby(level=0).mean().mean()
        
        return rolling_corr.iloc[-1], avg_pairwise_corr.iloc[-1]
        
    def calculate_hedge_ratio(self, long_short_corr, avg_correlation):
        """
        Calculate required hedge ratio based on correlations
        """
        if long_short_corr is None:
            return 0
            
        # Base hedge on long-short correlation
        base_hedge = max(0, (abs(long_short_corr) - self.correlation_threshold) / (1 - self.correlation_threshold))
        
        # Adjust based on average pairwise correlation
        correlation_adjustment = max(0, (avg_correlation - 0.5) / 0.5)
        
        # Combine and apply limits
        hedge_ratio = min(
            self.max_hedge_ratio,
            max(
                0,
                base_hedge * (1 + correlation_adjustment)
            )
        )
        
        return hedge_ratio if hedge_ratio > self.min_hedge_ratio else 0
        
    def update_hedges(self, data, index_data):
        """
        Update portfolio hedges based on correlation analysis
        """
        # Calculate correlations
        long_short_corr, avg_correlation = self.calculate_portfolio_correlations(data)
        
        # Calculate required hedge ratio
        required_hedge = self.calculate_hedge_ratio(long_short_corr, avg_correlation)
        
        # Get current portfolio value and exposure
        portfolio_value = self.get_portfolio_value()
        
        long_exposure = sum(p['position_size'] for p in self.positions.values() if p['direction'] == 'long')
        short_exposure = sum(p['position_size'] for p in self.positions.values() if p['direction'] == 'short')
        net_exposure = long_exposure - short_exposure
        
        # Calculate required hedge position
        required_hedge_size = net_exposure * required_hedge
        
        # Update index hedge position
        if required_hedge_size > 0:
            if 'INDEX' not in self.hedge_positions:
                # Initialize new hedge
                index_price = index_data['Close'].iloc[-1]
                hedge_units = required_hedge_size / index_price
                
                self.hedge_positions['INDEX'] = {
                    'units': hedge_units,
                    'entry_price': index_price,
                    'position_size': required_hedge_size,
                    'direction': 'short' if net_exposure > 0 else 'long'
                }
            else:
                # Adjust existing hedge
                current_hedge = self.hedge_positions['INDEX']
                index_price = index_data['Close'].iloc[-1]
                
                # Calculate hedge adjustment
                current_hedge_size = current_hedge['units'] * index_price
                hedge_difference = required_hedge_size - current_hedge_size
                
                # Apply gradual adjustment
                adjustment_size = hedge_difference * self.hedge_adjustment_factor
                new_units = current_hedge['units'] + (adjustment_size / index_price)
                
                self.hedge_positions['INDEX'] = {
                    'units': new_units,
                    'entry_price': current_hedge['entry_price'],
                    'position_size': new_units * index_price,
                    'direction': 'short' if net_exposure > 0 else 'long'
                }
        elif 'INDEX' in self.hedge_positions:
            # Remove hedge if no longer needed
            del self.hedge_positions['INDEX']
            
    def update_positions(self, data, current_date):
        """
        Update positions with current market data
        """
        for ticker, position in list(self.positions.items()):
            current_price = data[ticker]['Close'].iloc[-1]
            position_value = position['shares'] * current_price
            
            # Calculate unrealized P&L
            if position['direction'] == 'long':
                position['unrealized_pnl'] = position_value - position['cost_basis']
            else:  # short
                position['unrealized_pnl'] = position['cost_basis'] - position_value
                
            # Check stop loss and take profit
            if position['direction'] == 'long':
                if current_price <= position['stop_loss']:
                    self._close_position(ticker, current_price, current_date, 'stop_loss')
                elif current_price >= position['take_profit']:
                    self._close_position(ticker, current_price, current_date, 'take_profit')
            else:  # short
                if current_price >= position['stop_loss']:
                    self._close_position(ticker, current_price, current_date, 'stop_loss')
                elif current_price <= position['take_profit']:
                    self._close_position(ticker, current_price, current_date, 'take_profit')
                    
    def _close_position(self, ticker, current_price, current_date, reason):
        """
        Close a position and record the trade
        """
        position = self.positions[ticker]
        
        # Calculate realized P&L
        if position['direction'] == 'long':
            realized_pnl = position['shares'] * (current_price - position['entry_price'])
        else:  # short
            realized_pnl = position['shares'] * (position['entry_price'] - current_price)
            
        # Update capital
        old_capital = self.current_capital
        self.current_capital += realized_pnl
        
        # Debug logging
        print(f"Closing {position['direction']} position in {ticker}")
        print(f"Entry price: {position['entry_price']:.2f}, Exit price: {current_price:.2f}")
        print(f"Shares: {position['shares']:.2f}, Realized P&L: {realized_pnl:.2f}")
        print(f"Capital before: {old_capital:.2f}, after: {self.current_capital:.2f}")
        
        # Record trade
        trade = {
            'ticker': ticker,
            'direction': position['direction'],
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': current_date,
            'exit_price': current_price,
            'shares': position['shares'],
            'pnl': realized_pnl,
            'reason': reason
        }
        self.trades.append(trade)
        
        # Remove position
        del self.positions[ticker]
        
    def get_portfolio_value(self):
        """
        Calculate total portfolio value
        """
        total_value = self.current_capital
        for ticker, position in self.positions.items():
            total_value += position['unrealized_pnl']
        return total_value
    
    def get_position_summary(self):
        """
        Get summary of current positions
        """
        summary = {
            'long_positions': {},
            'short_positions': {},
            'total_positions': len(self.positions),
            'current_capital': self.current_capital
        }
        
        for ticker, position in self.positions.items():
            if position['direction'] == 'long':
                summary['long_positions'][ticker] = position
            else:
                summary['short_positions'][ticker] = position
                
        return summary

    def reset(self, initial_capital=None):
        """
        Reset the portfolio to initial state
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            self.min_capital = initial_capital * 0.85  # Reset minimum capital threshold
        self.current_capital = self.initial_capital
        self.positions = {}
        self.hedge_positions = {}
        self.trades = []
        
        print(f"Portfolio reset. New initial capital: {self.initial_capital:.2f}, Min capital: {self.min_capital:.2f}")
        
    def get_trades_for_date(self, date):
        """
        Get trades executed on a specific date
        """
        return [trade for trade in self.trades if trade['exit_date'] == date]

    def close_all_positions(self, data, current_date):
        """
        Close all open positions
        """
        for ticker in list(self.positions.keys()):
            current_price = data[ticker]['Close'].iloc[-1]
            self._close_position(ticker, current_price, current_date, 'close_all') 