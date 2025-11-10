"""
Data fetcher module - fetches stock data from various sources
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch stock data from Yahoo Finance"""
    
    def __init__(self):
        self.config = Config()
    
    def get_nifty500_symbols(self) -> List[str]:
        """Get list of Nifty 500 stock symbols"""
        try:
            # Try to read from local file first
            if os.path.exists(self.config.NIFTY500_FILE):
                df = pd.read_csv(self.config.NIFTY500_FILE)
                logger.info(f"Loaded {len(df)} stocks from local Nifty 500 file")
            else:
                # Download from NSE
                df = pd.read_csv(self.config.NIFTY_500_URL)
                # Save for future use
                os.makedirs(os.path.dirname(self.config.NIFTY500_FILE), exist_ok=True)
                df.to_csv(self.config.NIFTY500_FILE, index=False)
                logger.info(f"Downloaded {len(df)} stocks from NSE")
            
            # Convert to Yahoo Finance format (add .NS suffix)
            symbols = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
            return symbols
        
        except Exception as e:
            logger.error(f"Error fetching Nifty 500 symbols: {e}")
            # Fallback to a smaller list for testing
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]
    
    def fetch_intraday_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for a stock
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch 1-minute data for last 5 days
            df = ticker.history(period=f"{days}d", interval="1m")
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Add some calculated columns
            df['Symbol'] = symbol.replace('.NS', '')
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None
    
    def fetch_daily_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch daily historical data
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
        
        Returns:
            DataFrame with daily OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                return None
            
            df['Symbol'] = symbol.replace('.NS', '')
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None
    
    def get_current_price_and_volume(self, symbol: str) -> Optional[Dict]:
        """
        Get current price, volume, and day's stats
        
        Returns:
            Dict with open, high, low, close, volume, prev_close
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get today's data
            today_data = ticker.history(period="1d", interval="1m")
            
            if today_data.empty:
                return None
            
            current_price = today_data['Close'].iloc[-1]
            day_open = today_data['Open'].iloc[0]
            day_high = today_data['High'].max()
            day_low = today_data['Low'].min()
            volume = today_data['Volume'].sum()
            
            # Previous close
            prev_data = ticker.history(period="2d", interval="1d")
            prev_close = prev_data['Close'].iloc[-2] if len(prev_data) >= 2 else day_open
            
            # Calculate metrics
            day_change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Calculate how close to day high
            high_proximity = ((current_price - day_low) / (day_high - day_low)) * 100 if day_high != day_low else 0
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'current_price': round(current_price, 2),
                'open': round(day_open, 2),
                'high': round(day_high, 2),
                'low': round(day_low, 2),
                'prev_close': round(prev_close, 2),
                'volume': int(volume),
                'day_change_pct': round(day_change_pct, 2),
                'high_proximity_pct': round(high_proximity, 2)
            }
        
        except Exception as e:
            logger.error(f"Error getting current data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        if df is None or df.empty:
            return df
        
        try:
            # EMAs
            df['EMA_20'] = df['Close'].ewm(span=self.config.TECHNICAL.ema_short, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=self.config.TECHNICAL.ema_medium, adjust=False).mean()
            
            # SMA
            df['SMA_200'] = df['Close'].rolling(window=self.config.TECHNICAL.sma_long).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.TECHNICAL.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.TECHNICAL.rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=self.config.TECHNICAL.macd_fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=self.config.TECHNICAL.macd_slow, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=self.config.TECHNICAL.macd_signal, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Volume average
            df['Volume_Avg_20'] = df['Volume'].rolling(window=self.config.TECHNICAL.volume_lookback).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_Avg_20']
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def fetch_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data for a stock
        
        Returns:
            Dict with market cap, PE, debt/equity, ROE, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'market_cap': info.get('marketCap', 0) / 10000000,  # in crores
                'pe_ratio': info.get('trailingPE', 0),
                'debt_to_equity': info.get('debtToEquity', 0) / 100,  # Convert to ratio
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
        
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None
    
    def batch_fetch_daily_data(self, symbols: List[str], days: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily historical data for multiple stocks at once using yf.download()
        This is much faster than fetching one by one

        Args:
            symbols: List of stock symbols
            days: Number of days of history

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logger.info(f"Batch fetching daily data for {len(symbols)} stocks...")

            # Use yf.download for batch fetch - much faster!
            import warnings
            import logging

            # Suppress yfinance logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    group_by='ticker',
                    threads=True,
                    progress=False
                )

            results = {}

            # Process results for each symbol
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        # Single symbol returns different structure
                        symbol_df = df.copy()
                    else:
                        # Multiple symbols - extract this symbol's data
                        symbol_df = df[symbol].copy()

                    if not symbol_df.empty:
                        symbol_df['Symbol'] = symbol.replace('.NS', '')
                        results[symbol] = symbol_df

                except (KeyError, AttributeError):
                    continue

            logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} stocks")
            return results

        except Exception as e:
            logger.error(f"Error in batch fetch: {e}")
            return {}

    def batch_fetch_current_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch current price and volume data for multiple stocks
        Uses batch download for efficiency

        Returns:
            Dict mapping symbol to current data dict
        """
        try:
            logger.info(f"Batch fetching current data for {len(symbols)} stocks...")

            # Download 2 days of data to get current and previous close
            import warnings
            import logging

            # Suppress yfinance logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    symbols,
                    period="2d",
                    interval="1d",
                    group_by='ticker',
                    threads=True,
                    progress=False
                )

            results = {}
            failed_symbols = []

            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        symbol_df = df.copy()
                    else:
                        symbol_df = df[symbol].copy()

                    if symbol_df.empty or len(symbol_df) < 1:
                        failed_symbols.append(symbol)
                        continue

                    # Get latest data
                    latest = symbol_df.iloc[-1]
                    prev_close = symbol_df.iloc[-2]['Close'] if len(symbol_df) >= 2 else latest['Open']

                    current_price = latest['Close']
                    day_change_pct = ((current_price - prev_close) / prev_close) * 100

                    # Calculate high proximity
                    high_proximity = ((current_price - latest['Low']) / (latest['High'] - latest['Low'])) * 100 if latest['High'] != latest['Low'] else 0

                    results[symbol] = {
                        'symbol': symbol.replace('.NS', ''),
                        'current_price': round(float(current_price), 2),
                        'open': round(float(latest['Open']), 2),
                        'high': round(float(latest['High']), 2),
                        'low': round(float(latest['Low']), 2),
                        'prev_close': round(float(prev_close), 2),
                        'volume': int(latest['Volume']),
                        'day_change_pct': round(float(day_change_pct), 2),
                        'high_proximity_pct': round(float(high_proximity), 2)
                    }

                except (KeyError, AttributeError, IndexError) as e:
                    failed_symbols.append(symbol)
                    continue

            logger.info(f"Successfully fetched current data for {len(results)}/{len(symbols)} stocks")
            if failed_symbols and len(failed_symbols) < 50:  # Only log if reasonable number
                logger.debug(f"Failed to fetch: {', '.join([s.replace('.NS', '') for s in failed_symbols[:10]])}{' ...' if len(failed_symbols) > 10 else ''}")

            return results

        except Exception as e:
            logger.error(f"Error in batch current data fetch: {e}")
            return {}

    def batch_fetch_fundamentals(self, symbols: List[str], batch_size: int = 50) -> Dict[str, Dict]:
        """
        Fetch fundamental data for multiple stocks
        Note: This still needs to be done one by one, but with batching and rate limiting

        Args:
            symbols: List of stock symbols
            batch_size: Number of symbols to process in each batch

        Returns:
            Dict mapping symbol to fundamentals dict
        """
        results = {}
        total = len(symbols)

        logger.info(f"Fetching fundamentals for {total} stocks in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")

            for symbol in batch:
                try:
                    # Suppress yfinance error logging
                    import logging
                    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    results[symbol] = {
                        'symbol': symbol.replace('.NS', ''),
                        'market_cap': info.get('marketCap', 0) / 10000000,
                        'pe_ratio': info.get('trailingPE', 0),
                        'debt_to_equity': info.get('debtToEquity', 0) / 100,
                        'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                    }

                    # Small delay between API calls
                    time.sleep(0.5)

                except Exception as e:
                    # Silently handle errors - provide default values
                    results[symbol] = {'sector': 'Unknown', 'market_cap': 0, 'pe_ratio': 0, 'debt_to_equity': 0, 'roe': 0, 'industry': 'Unknown'}
                    continue

            # Delay between batches
            if i + batch_size < total:
                time.sleep(2.0)

        logger.info(f"Fetched fundamentals for {len(results)}/{total} stocks")
        return results

    def batch_fetch_with_rate_limit(self, symbols: List[str], fetch_func, **kwargs):
        """
        Fetch data for multiple symbols with rate limiting
        DEPRECATED: Use batch_fetch_daily_data or batch_fetch_current_data instead

        Args:
            symbols: List of stock symbols
            fetch_func: Function to call for each symbol
            **kwargs: Additional arguments to pass to fetch_func

        Returns:
            Dict mapping symbol to result
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching {symbol} ({i+1}/{total})")

            result = fetch_func(symbol, **kwargs)
            if result is not None:
                results[symbol] = result

            # Rate limiting
            if i < total - 1:  # Don't sleep after last item
                time.sleep(self.config.RATE_LIMIT_DELAY)

        return results


# Standalone functions for easy imports
def get_stock_list() -> List[str]:
    """Get Nifty 500 stock list"""
    fetcher = DataFetcher()
    return fetcher.get_nifty500_symbols()


def get_current_data(symbol: str) -> Optional[Dict]:
    """Get current price and volume data"""
    fetcher = DataFetcher()
    return fetcher.get_current_price_and_volume(symbol)


def get_historical_data(symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
    """Get historical daily data with indicators"""
    fetcher = DataFetcher()
    df = fetcher.fetch_daily_data(symbol, days)
    if df is not None:
        df = fetcher.calculate_technical_indicators(df)
    return df
