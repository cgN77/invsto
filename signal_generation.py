import pandas as pd
import numpy as np
from scipy import stats

class SignalGenerator:
    def __init__(self):
        # Signal parameters
        self.momentum_window = 20
        self.volatility_window = 20
        self.rsi_window = 14
        self.signal_decay = 0.5
        
    def calculate_momentum_signal(self, data):
        """
        Calculate momentum signal using risk-adjusted returns
        """
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Calculate volatility
        volatility = returns.rolling(window=self.momentum_window).std()
        
        # Calculate risk-adjusted momentum
        momentum = returns.rolling(window=self.momentum_window).mean() / (volatility + 1e-6)
        
        # Add trend persistence
        trend = (data['Close'] > data['Close'].shift(1)).astype(int).rolling(window=self.momentum_window).mean()
        
        # Combine signals
        signal = momentum * trend
        
        # Normalize signal
        signal = (signal - signal.rolling(window=60).mean()) / (signal.rolling(window=60).std() + 1e-6)
        
        return signal.clip(-3, 3)  # Clip extreme values
        
    def calculate_mean_reversion_signal(self, data):
        """
        Calculate mean reversion signal with volume confirmation
        """
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate price position
        rolling_high = data['Close'].rolling(window=20).max()
        rolling_low = data['Close'].rolling(window=20).min()
        price_position = (data['Close'] - rolling_low) / (rolling_high - rolling_low + 1e-6)
        
        # Calculate VVR (Volume-Volatility Ratio)
        volume = data['Volume'] if 'Volume' in data else pd.Series(1, index=data.index)
        avg_volume = volume.rolling(window=20).mean()
        price_std = data['Close'].rolling(window=20).std()
        vvr = avg_volume / (price_std + 1e-6)
        
        # Normalize VVR
        vvr_zscore = (vvr - vvr.rolling(window=60).mean()) / (vvr.rolling(window=60).std() + 1e-6)
        
        # Combine signals
        signal = (
            0.5 * (0.5 - price_position) * np.sign(vvr_zscore) +  # Price position with volume confirmation
            0.3 * (60 - rsi) / 60 +  # RSI signal
            0.2 * vvr_zscore  # Raw VVR signal
        )
        
        return signal.clip(-3, 3)  # Clip extreme values
        
    def calculate_volatility_signal(self, data):
        """
        Calculate volatility signal using both historical and implied volatility
        """
        # Calculate historical volatility
        returns = data['Close'].pct_change()
        hv_20 = returns.rolling(window=20).std() * np.sqrt(252)
        hv_60 = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Calculate HV trend
        hv_trend = (hv_20 - hv_60) / (hv_60 + 1e-6)
        
        # Normalize HV
        hv_zscore = (hv_20 - hv_20.rolling(window=252).mean()) / (hv_20.rolling(window=252).std() + 1e-6)
        
        # Calculate final volatility score
        vol_score = -hv_zscore  # Negative because high volatility = lower position size
        
        # Adjust for price trend
        price_trend = data['Close'].pct_change(20)
        trend_adjustment = np.sign(price_trend) * np.minimum(abs(price_trend), 0.1)
        
        # Combine signals
        signal = vol_score + trend_adjustment
        
        return signal.clip(-3, 3)  # Clip extreme values
        
    def generate_signals(self, data):
        """
        Generate trading signals
        """
        # Calculate individual signals for each stock
        signals = {}
        composite_scores = {}
        
        for ticker, stock_data in data.items():
            momentum = self.calculate_momentum_signal(stock_data)
            mean_reversion = self.calculate_mean_reversion_signal(stock_data)
            volatility = self.calculate_volatility_signal(stock_data)
            
            # Calculate composite signal
            composite = (
                momentum.iloc[-1] +
                mean_reversion.iloc[-1] +
                volatility.iloc[-1]
            ) / 3
            
            signals[ticker] = {
                'momentum': momentum.iloc[-1],
                'mean_reversion': mean_reversion.iloc[-1],
                'volatility': volatility.iloc[-1],
                'composite': composite
            }
            composite_scores[ticker] = composite
            
        return {
            'signals': signals,
            'composite': composite_scores
        } 