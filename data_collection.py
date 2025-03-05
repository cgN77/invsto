import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.stocks = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'GOOGL': 'Google',
            'META': 'Facebook',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan Chase',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart'
        }
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch historical data for all stocks
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        data = {}
        for ticker in self.stocks.keys():
            try:
                stock = yf.Ticker(ticker)
                # Get historical data
                hist_data = stock.history(start=start_date, end=end_date)
                
                # Get fundamental data
                info = stock.info
                fundamentals = {
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('forwardPE', None),
                    'dividend_yield': info.get('dividendYield', None),
                    'beta': info.get('beta', None)
                }
                
                data[ticker] = {
                    'historical': hist_data,
                    'fundamentals': fundamentals
                }
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                
        return data
    
    def prepare_data(self, data):
        """
        Prepare data for analysis by calculating returns and technical indicators
        """
        processed_data = {}
        
        for ticker, stock_data in data.items():
            hist_data = stock_data['historical']
            
            # Calculate daily returns
            hist_data['Returns'] = hist_data['Close'].pct_change()
            
            # Calculate moving averages
            hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
            
            # Calculate volatility (30-day rolling standard deviation of returns)
            hist_data['Volatility'] = hist_data['Returns'].rolling(window=30).std() * np.sqrt(252)
            
            # Calculate momentum (20-day returns)
            hist_data['Momentum'] = hist_data['Close'].pct_change(periods=20)
            
            processed_data[ticker] = {
                'historical': hist_data,
                'fundamentals': stock_data['fundamentals']
            }
            
        return processed_data
    
    def get_benchmark_data(self, start_date=None, end_date=None):
        """
        Fetch S&P 500 data for benchmark comparison
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            sp500 = yf.Ticker('^GSPC')
            benchmark_data = sp500.history(start=start_date, end=end_date)
            benchmark_data['Returns'] = benchmark_data['Close'].pct_change()
            return benchmark_data
        except Exception as e:
            print(f"Error fetching S&P 500 data: {str(e)}")
            return None 