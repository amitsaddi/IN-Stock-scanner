"""
Nifty 500 Historical Data Fetcher
Downloads 6 months of historical data for Nifty 500 stocks and loads into SQLite database
Optimized with:
- Parallel processing with ThreadPoolExecutor
- Bulk yfinance downloads
- Smart rate limiting
- Progress checkpointing
- Bulk database inserts
- Thread-safe operations
"""
import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
from functools import wraps
import sqlite3

# Add parent directories to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "IN-testing" / "scripts"))

from db_schema import NiftyDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(script_dir.parent / "logs" / f"data_fetcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter for API calls
    Limits to specified calls per minute with automatic throttling
    """
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter

        Args:
            calls_per_minute: Maximum API calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]

            if len(self.calls) >= self.calls_per_minute:
                # Wait until oldest call expires
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                logger.debug(f"Rate limit reached. Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # Clear old calls after waiting
                now = time.time()
                self.calls = [t for t in self.calls if now - t < 60]

            self.calls.append(now)


# Nifty 500 representative stocks for testing
NIFTY_500_TEST_STOCKS = [
    "RELIANCE.NS",     # Energy
    "TCS.NS",          # IT
    "HDFCBANK.NS",     # Banking
    "INFY.NS",         # IT
    "ICICIBANK.NS",    # Banking
    "HINDUNILVR.NS",   # FMCG
    "ITC.NS",          # FMCG
    "SBIN.NS",         # PSU Bank
    "BHARTIARTL.NS",   # Telecom
    "LT.NS"            # Capital Goods
]

# Top 200 Nifty 500 stocks by market cap (hardcoded for reliable operation)
NIFTY_500_TOP_200 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
    "ADANIENT.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "M&M.NS",
    "HCLTECH.NS", "TECHM.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "INDUSINDBK.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "ADANIPORTS.NS", "GRASIM.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS",
    "HEROMOTOCO.NS", "VEDL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
    "BPCL.NS", "IOC.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "SIEMENS.NS",
    "DLF.NS", "PIDILITIND.NS", "HAVELLS.NS", "AMBUJACEM.NS", "ACC.NS",
    "GAIL.NS", "BOSCHLTD.NS", "SHREECEM.NS", "UPL.NS", "BERGEPAINT.NS",
    "COLPAL.NS", "MARICO.NS", "INDIGO.NS", "BANKBARODA.NS", "PNB.NS",
    "CANBK.NS", "UNIONBANK.NS", "IDBI.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
    "PFC.NS", "RECLTD.NS", "IRCTC.NS", "HAL.NS", "BEL.NS",
    "BHEL.NS", "SAIL.NS", "NMDC.NS", "RVNL.NS", "IRFC.NS",
    "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "DMART.NS",
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "PHOENIXLTD.NS",
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "ICICIGI.NS", "SBICARD.NS", "BAJAJHLDNG.NS",
    "TORNTPHARM.NS", "BIOCON.NS", "ALKEM.NS", "LUPIN.NS", "AUROPHARMA.NS",
    # Additional top 100 stocks
    "TVSMOTOR.NS", "MOTHERSON.NS", "BALKRISIND.NS", "MRF.NS", "APOLLOTYRE.NS",
    "ATUL.NS", "DEEPAKNTR.NS", "SRF.NS", "AARTI.NS", "GNFC.NS",
    "PIDILITIND.NS", "KANSAINER.NS", "VOLTAS.NS", "BLUEDART.NS", "VBL.NS",
    "TATAELXSI.NS", "PERSISTENT.NS", "COFORGE.NS", "LTIM.NS", "MPHASIS.NS",
    "INOXWIND.NS", "SUZLON.NS", "TATAPOWER.NS", "NHPC.NS", "SJVN.NS",
    "CUMMINSIND.NS", "ABB.NS", "THERMAX.NS", "CROMPTON.NS", "POLYCAB.NS",
    "APLAPOLLO.NS", "JINDALSTEL.NS", "RATNAMANI.NS", "TIINDIA.NS", "WELCORP.NS",
    "ZYDUSLIFE.NS", "LAURUSLABS.NS", "GRANULES.NS", "SYNGENE.NS", "NATCOPHARM.NS",
    "DIXON.NS", "AMBER.NS", "SYMPHONY.NS", "WHIRLPOOL.NS", "VGUARD.NS",
    "BATAINDIA.NS", "RELAXO.NS", "PAGEIND.NS", "JUBLFOOD.NS", "WESTLIFE.NS",
    "IRCTC.NS", "IRCON.NS", "CONCOR.NS", "GICRE.NS", "STARHEALTH.NS",
    "PGHH.NS", "GILLETTE.NS", "HONAUT.NS", "3MINDIA.NS", "ABBOTINDIA.NS",
    "SANOFI.NS", "PFIZER.NS", "GSK.NS", "GLAXO.NS", "ALEMBICLTD.NS",
    "LALPATHLAB.NS", "METROPOLIS.NS", "THYROCARE.NS", "KIMS.NS", "RAINBOWHSP.NS",
    "GRINDWELL.NS", "CARBORUNIV.NS", "STARCEMENT.NS", "JKLAKSHMI.NS", "RAMCOCEM.NS",
    "ORIENTCEM.NS", "JKCEMENT.NS", "HEIDELBERG.NS", "PRSMJOHNSN.NS", "FINEORG.NS"
]


class Nifty500DataFetcher:
    """Fetch and process historical data for Nifty 500 stocks"""

    def __init__(self, db_manager: NiftyDatabaseManager, days: int = 180,
                 workers: int = 5, force: bool = False):
        """
        Initialize data fetcher

        Args:
            db_manager: Database manager instance (used for main thread and to get db_path)
            days: Number of days of historical data to fetch (default: 180)
            workers: Number of parallel workers for concurrent downloads (default: 5)
            force: Force full re-download even if data exists (default: False)
        """
        self.db = db_manager  # Main thread database connection
        self.db_path = db_manager.db_path  # Store path for thread-local connections
        self.days = days
        self.workers = workers
        self.force = force
        self.successful_downloads = []
        self.failed_downloads = []
        self.skipped_stocks = []
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.db_lock = Lock()  # Thread-safe database operations
        self._local = threading.local()  # Thread-local storage for database connections

    def get_db_connection(self) -> NiftyDatabaseManager:
        """
        Get thread-local database connection.
        Each thread gets its own connection to avoid SQLite threading issues.

        Returns:
            Database manager instance for the current thread
        """
        if not hasattr(self._local, 'db'):
            # Create new connection for this thread
            logger.debug(f"Creating thread-local DB connection for thread {threading.current_thread().ident}")
            self._local.db = NiftyDatabaseManager(self.db_path)
            self._local.db.connect()
        return self._local.db

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        if df is None or df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical indicators")
            return df

        try:
            # Moving Averages
            df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['sma_200'] = df['Close'].rolling(window=200, min_periods=50).mean()

            # RSI (14-period)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD (12, 26, 9)
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # ATR (14-period)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_percent'] = (df['atr'] / df['Close']) * 100

            # Bollinger Bands (20, 2)
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # Volume metrics
            df['volume_20d_avg'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_20d_avg']

            # 52-week high/low (252 trading days, min 50 days)
            df['week_52_high'] = df['Close'].rolling(window=252, min_periods=50).max()
            df['week_52_low'] = df['Close'].rolling(window=252, min_periods=50).min()
            df['week_52_high_proximity'] = (df['Close'] / df['week_52_high']) * 100

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single stock

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            DataFrame with OHLCV data and indicators
        """
        try:
            logger.info(f"Fetching data for {symbol}...")

            # Suppress yfinance warnings
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)

            ticker = yf.Ticker(symbol)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 30)  # Extra buffer for indicators

            # Fetch data
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            # Restore logging level
            yf_logger.setLevel(original_level)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return None

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)

            # Keep only last 'days' worth of data
            df = df.tail(self.days)

            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a stock

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            Dict with fundamental data
        """
        try:
            # Suppress yfinance warnings
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Restore logging level
            yf_logger.setLevel(original_level)

            # Market cap in crores (INR 10,000,000 = 1 crore)
            market_cap_raw = info.get('marketCap', 0)
            market_cap = market_cap_raw / 10000000 if market_cap_raw else 0

            # Get sector
            sector = info.get('sector', 'Unknown')

            # Debt to equity ratio (convert from percentage to ratio)
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0

            # ROE (convert to percentage if needed)
            roe_raw = info.get('returnOnEquity', 0)
            roe = roe_raw * 100 if roe_raw and roe_raw < 1 else roe_raw if roe_raw else 0

            return {
                'sector': sector,
                'market_cap': market_cap,
                'debt_to_equity': debt_to_equity,
                'roe': roe
            }

        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
            return {
                'sector': 'Unknown',
                'market_cap': 0,
                'debt_to_equity': 0,
                'roe': 0
            }

    def get_last_date_in_db(self, symbol: str) -> Optional[datetime]:
        """
        Get the last date for which we have data for a stock

        Args:
            symbol: Stock symbol (without .NS suffix)

        Returns:
            Last date in database or None if no data exists
        """
        try:
            table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"
            db = self.get_db_connection()  # Get thread-local connection
            with self.db_lock:
                db.cursor.execute(
                    f"SELECT MAX(date) FROM {table_name}"
                )
                result = db.cursor.fetchone()

            if result and result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d')
            return None

        except Exception as e:
            logger.debug(f"No existing data for {symbol}: {e}")
            return None

    def download_bulk(self, symbols: List[str], start_date: datetime,
                     end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Download multiple stocks in one bulk call using yfinance

        Args:
            symbols: List of stock symbols (with .NS suffix)
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()

            # Suppress yfinance warnings
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)

            # Download all symbols at once
            symbols_str = " ".join(symbols)
            logger.info(f"Bulk downloading {len(symbols)} stocks...")

            data = yf.download(
                symbols_str,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False,
                threads=True
            )

            # Restore logging level
            yf_logger.setLevel(original_level)

            # Parse the multi-index DataFrame
            result = {}

            if len(symbols) == 1:
                # Single stock - different structure
                if not data.empty:
                    result[symbols[0]] = data
            else:
                # Multiple stocks - multi-index structure
                for symbol in symbols:
                    try:
                        if symbol in data.columns.levels[0]:
                            df = data[symbol]
                            if not df.empty:
                                result[symbol] = df
                    except (KeyError, AttributeError):
                        logger.warning(f"No data returned for {symbol}")

            logger.info(f"Successfully downloaded {len(result)}/{len(symbols)} stocks")
            return result

        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            return {}

    def store_stock_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Store stock data in database using bulk inserts

        Args:
            symbol: Stock symbol (without .NS suffix)
            df: DataFrame with OHLCV and indicator data

        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = f"stock_{symbol.replace('.', '_').replace('-', '_').replace('&', '_')}"

            # Prepare DataFrame for bulk insert
            df_to_insert = pd.DataFrame({
                'date': df.index.strftime('%Y-%m-%d'),
                'open': df['Open'].astype(float),
                'high': df['High'].astype(float),
                'low': df['Low'].astype(float),
                'close': df['Close'].astype(float),
                'volume': df['Volume'].astype(int),
                'adj_close': df.get('Close', df['Close']).astype(float),
                'ema_20': df['ema_20'].astype(float) if 'ema_20' in df else None,
                'ema_50': df['ema_50'].astype(float) if 'ema_50' in df else None,
                'sma_200': df['sma_200'].astype(float) if 'sma_200' in df else None,
                'rsi': df['rsi'].astype(float) if 'rsi' in df else None,
                'macd': df['macd'].astype(float) if 'macd' in df else None,
                'macd_signal': df['macd_signal'].astype(float) if 'macd_signal' in df else None,
                'macd_hist': df['macd_hist'].astype(float) if 'macd_hist' in df else None,
                'atr': df['atr'].astype(float) if 'atr' in df else None,
                'atr_percent': df['atr_percent'].astype(float) if 'atr_percent' in df else None,
                'bb_upper': df['bb_upper'].astype(float) if 'bb_upper' in df else None,
                'bb_middle': df['bb_middle'].astype(float) if 'bb_middle' in df else None,
                'bb_lower': df['bb_lower'].astype(float) if 'bb_lower' in df else None,
                'volume_20d_avg': df['volume_20d_avg'].astype(float) if 'volume_20d_avg' in df else None,
                'volume_ratio': df['volume_ratio'].astype(float) if 'volume_ratio' in df else None,
                'week_52_high': df['week_52_high'].astype(float) if 'week_52_high' in df else None,
                'week_52_low': df['week_52_low'].astype(float) if 'week_52_low' in df else None,
                'week_52_high_proximity': df['week_52_high_proximity'].astype(float) if 'week_52_high_proximity' in df else None,
            })

            # Get thread-local connection and use bulk insert with thread-safe lock
            db = self.get_db_connection()
            with self.db_lock:
                # Delete any existing data in the date range to prevent duplicates
                first_date_str = df_to_insert['date'].iloc[0]
                last_date_str = df_to_insert['date'].iloc[-1]

                db.cursor.execute(f"""
                    DELETE FROM {table_name}
                    WHERE date BETWEEN ? AND ?
                """, (first_date_str, last_date_str))

                # Insert new data
                df_to_insert.to_sql(
                    table_name,
                    db.conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
                db.conn.commit()

            logger.info(f"Stored {len(df_to_insert)} records for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            return False

    def check_if_downloaded(self, symbol: str) -> Tuple[bool, Optional[datetime]]:
        """
        Check if stock data already exists in database and get last date

        Args:
            symbol: Stock symbol (without .NS suffix)

        Returns:
            Tuple of (needs_download, last_date)
            - needs_download: False if complete and data is from today
            - last_date: Last date in database or None
        """
        try:
            # If force mode, always download
            if self.force:
                return (True, None)

            db = self.get_db_connection()  # Get thread-local connection
            with self.db_lock:
                db.cursor.execute(
                    "SELECT data_status, total_records FROM stock_metadata WHERE symbol = ?",
                    (symbol,)
                )
                result = db.cursor.fetchone()

            # Check if complete
            if result and result[0] == 'complete' and result[1] and result[1] > 0:
                # Get last date for incremental update
                last_date = self.get_last_date_in_db(symbol)

                # Skip only if data is from today
                if last_date:
                    days_old = (datetime.now() - last_date).days
                    if days_old == 0:
                        logger.info(f"Skipping {symbol} - data is already up to date ({result[1]} records, last: {last_date.strftime('%Y-%m-%d')})")
                        return (False, last_date)
                    else:
                        logger.info(f"Updating {symbol} incrementally - data is {days_old} days old, last: {last_date.strftime('%Y-%m-%d')}")
                        return (True, last_date)

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking download status for {symbol}: {e}")
            return (True, None)

    def get_incremental_date_range(self, symbol: str, last_date: Optional[datetime]) -> Tuple[datetime, datetime, Optional[datetime]]:
        """
        Calculate date range for incremental update and pruning

        Args:
            symbol: Stock symbol
            last_date: Last date in database (or None for new stock)

        Returns:
            Tuple of (start_date, end_date, prune_before_date)
            - start_date: Date to start downloading from
            - end_date: Date to download until (today)
            - prune_before_date: Delete records before this date (to maintain rolling window)
        """
        end_date = datetime.now()

        if last_date:
            # Incremental update: start from day after last recorded date
            start_date = last_date + timedelta(days=1)

            # Calculate rolling window cutoff (maintain 180 days)
            window_start = end_date - timedelta(days=self.days)
            prune_before = window_start if last_date < window_start else None

            return (start_date, end_date, prune_before)
        else:
            # New stock: download full period
            start_date = end_date - timedelta(days=self.days + 30)
            return (start_date, end_date, None)

    def process_bulk_batch(self, symbols: List[str], skip_existing: bool = True) -> int:
        """
        Process a batch of stocks using bulk download with incremental updates

        Args:
            symbols: List of stock symbols (with .NS suffix)
            skip_existing: If True, skip stocks that are already downloaded

        Returns:
            Number of successfully processed stocks
        """
        if not symbols:
            return 0

        success_count = 0

        try:
            # Categorize stocks: skip, incremental update, or full download
            stocks_to_skip = []
            stocks_for_incremental = {}  # symbol -> last_date
            stocks_for_full_download = []

            for symbol_with_suffix in symbols:
                symbol = symbol_with_suffix.replace('.NS', '')
                if skip_existing:
                    needs_download, last_date = self.check_if_downloaded(symbol)
                    if not needs_download:
                        stocks_to_skip.append(symbol)
                        self.skipped_stocks.append(symbol)
                        continue

                    if last_date:
                        # Incremental update
                        stocks_for_incremental[symbol_with_suffix] = last_date
                    else:
                        # New stock - full download
                        stocks_for_full_download.append(symbol_with_suffix)
                else:
                    stocks_for_full_download.append(symbol_with_suffix)

            if stocks_to_skip:
                logger.info(f"Skipping {len(stocks_to_skip)} up-to-date stocks")

            # Process full downloads (new stocks)
            if stocks_for_full_download:
                logger.info(f"Full download for {len(stocks_for_full_download)} new stocks...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.days + 30)

                bulk_data = self.download_bulk(stocks_for_full_download, start_date, end_date)
                success_count += self._process_downloaded_data(bulk_data, is_incremental=False)

            # Process incremental updates
            if stocks_for_incremental:
                logger.info(f"Incremental update for {len(stocks_for_incremental)} existing stocks...")

                # Group by similar date ranges for efficient bulk downloads
                date_groups = {}
                for symbol_with_suffix, last_date in stocks_for_incremental.items():
                    # Group by last_date to batch similar updates
                    date_key = last_date.strftime('%Y-%m-%d')
                    if date_key not in date_groups:
                        date_groups[date_key] = []
                    date_groups[date_key].append((symbol_with_suffix, last_date))

                # Process each date group
                for date_key, group in date_groups.items():
                    symbols_list = [s for s, _ in group]
                    last_date = group[0][1]  # All have same date

                    start_date = last_date + timedelta(days=1)
                    end_date = datetime.now()

                    logger.info(f"Updating {len(symbols_list)} stocks from {start_date.strftime('%Y-%m-%d')}...")
                    bulk_data = self.download_bulk(symbols_list, start_date, end_date)
                    success_count += self._process_downloaded_data(bulk_data, is_incremental=True)

            return success_count

        except Exception as e:
            logger.error(f"Error in bulk batch processing: {e}")
            return success_count

    def _process_downloaded_data(self, bulk_data: Dict[str, pd.DataFrame], is_incremental: bool = False) -> int:
        """
        Process downloaded stock data (helper method)

        Args:
            bulk_data: Dictionary mapping symbols to DataFrames
            is_incremental: If True, only keep new data (not full 180 days)

        Returns:
            Number of successfully processed stocks
        """
        success_count = 0
        db = self.get_db_connection()

        for symbol_with_suffix, df in bulk_data.items():
            symbol = symbol_with_suffix.replace('.NS', '')

            try:
                # Mark as downloading
                with self.db_lock:
                    db.update_metadata(symbol, data_status='downloading')

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    self.failed_downloads.append(symbol)
                    with self.db_lock:
                        db.update_metadata(symbol, data_status='failed', total_records=0)
                    continue

                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)

                # For full download, keep only requested days; for incremental, keep all new data
                if not is_incremental:
                    df = df.tail(self.days)

                if df.empty:
                    logger.warning(f"Empty data after processing for {symbol}")
                    self.failed_downloads.append(symbol)
                    with self.db_lock:
                        db.update_metadata(symbol, data_status='failed', total_records=0)
                    continue

                # Store in database (will delete duplicates automatically)
                if not self.store_stock_data(symbol, df):
                    self.failed_downloads.append(symbol)
                    with self.db_lock:
                        db.update_metadata(symbol, data_status='failed', total_records=0)
                    continue

                # Fetch fundamentals (only needed for new stocks or periodically)
                fundamentals = self.fetch_fundamentals(symbol_with_suffix)

                # Update metadata
                first_date = df.index[0].strftime('%Y-%m-%d')
                last_date = df.index[-1].strftime('%Y-%m-%d')

                with self.db_lock:
                    db.update_metadata(
                        symbol,
                        sector=fundamentals['sector'],
                        market_cap=fundamentals['market_cap'],
                        debt_to_equity=fundamentals['debt_to_equity'],
                        roe=fundamentals['roe'],
                        first_date=first_date,
                        last_date=last_date,
                        total_records=len(df),
                        data_status='complete'
                    )

                self.successful_downloads.append(symbol)
                success_count += 1
                logger.info(f"Successfully processed {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.failed_downloads.append(symbol)
                with self.db_lock:
                    db.update_metadata(symbol, data_status='failed', total_records=0)

        return success_count

    def process_batch_parallel(self, symbols: List[str], skip_existing: bool = True) -> None:
        """
        Process stocks in parallel using ThreadPoolExecutor

        Args:
            symbols: List of stock symbols (with .NS suffix)
            skip_existing: If True, skip stocks that are already downloaded
        """
        total = len(symbols)
        logger.info(f"Starting parallel processing of {total} stocks with {self.workers} workers")

        # Split symbols into batches for bulk download
        bulk_batch_size = 25  # Process 25 stocks at a time in bulk
        completed = 0
        start_time = time.time()

        # Process in bulk batches with parallel workers
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}

            for i in range(0, total, bulk_batch_size):
                batch = symbols[i:i + bulk_batch_size]
                future = executor.submit(self.process_bulk_batch, batch, skip_existing)
                futures[future] = batch

            # Collect results
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    success_count = future.result()
                    completed += len(batch)
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0

                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) - "
                               f"Rate: {rate:.1f} stocks/sec - ETA: {eta/60:.1f} min")

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("PARALLEL DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Total stocks: {total}")
        logger.info(f"Successful: {len(self.successful_downloads)}")
        logger.info(f"Skipped (already downloaded): {len(self.skipped_stocks)}")
        logger.info(f"Failed: {len(self.failed_downloads)}")
        logger.info(f"Average rate: {total/elapsed:.2f} stocks/second")

        if self.failed_downloads:
            logger.warning(f"Failed stocks: {', '.join(self.failed_downloads[:20])}")
            if len(self.failed_downloads) > 20:
                logger.warning(f"... and {len(self.failed_downloads) - 20} more")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Fetch Nifty 500 historical data with parallel optimization'
    )
    parser.add_argument('--symbols', nargs='+', help='Custom list of symbols (e.g., RELIANCE.NS TCS.NS)')
    parser.add_argument('--days', type=int, default=180, help='Number of days of history (default: 180)')
    parser.add_argument('--test', action='store_true', help='Test mode - download only 10 stocks')
    parser.add_argument('--db-path', type=str, help='Custom database path')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--force', action='store_true', help='Force full re-download even if data exists')

    args = parser.parse_args()

    # Ensure log directory exists
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Get symbols
    if args.symbols:
        # Custom symbols provided
        symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in args.symbols]
        logger.info(f"Using custom symbol list: {len(symbols)} stocks")
    elif args.test:
        # Test mode - 10 stocks
        symbols = NIFTY_500_TEST_STOCKS
        logger.info(f"TEST MODE: Using 10 diverse Nifty 500 stocks")
    else:
        # Full Nifty 500 list (top 200 for now)
        symbols = NIFTY_500_TOP_200
        logger.info(f"Using top 200 Nifty 500 stocks: {len(symbols)} stocks")

    # Initialize database
    db_path = args.db_path if args.db_path else None
    db = NiftyDatabaseManager(db_path)

    try:
        db.connect()
        logger.info(f"Connected to database: {db.db_path}")

        # Create metadata table
        db.create_metadata_table()

        # Create tables for all symbols
        logger.info("Creating stock tables...")
        for symbol in symbols:
            clean_symbol = symbol.replace('.NS', '')
            db.create_stock_table(clean_symbol)

        # Initialize data fetcher with new parameters
        fetcher = Nifty500DataFetcher(
            db,
            days=args.days,
            workers=args.workers,
            force=args.force
        )

        # Process stocks using parallel mode
        logger.info(f"Using PARALLEL mode with {args.workers} workers")
        fetcher.process_batch_parallel(
            symbols,
            skip_existing=not args.force
        )

        # Create unified view
        logger.info("Creating unified view...")
        db.create_unified_view()

        # Optimize database
        logger.info("Optimizing database...")
        db.vacuum()

        logger.info("=" * 80)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Database location: {db.db_path}")
        logger.info(f"Total stocks in database: {db.get_stock_count()}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
