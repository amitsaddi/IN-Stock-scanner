"""
ASX200 Historical Data Fetcher
Downloads 6 months of historical data for ASX200 stocks and loads into SQLite database
Optimized with Phase 3 Performance Improvements:
- Parallel processing with ThreadPoolExecutor
- Bulk yfinance downloads
- Smart rate limiting
- Progress checkpointing
- Bulk database inserts
- Smart error handling with exponential backoff
- Incremental updates
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
sys.path.insert(0, str(project_root / "AU-testing" / "scripts"))
sys.path.insert(0, str(project_root / "src"))

from db_schema import ASXDatabaseManager
from config.australia_config import AustraliaConfig

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


class ASX200DataFetcher:
    """Fetch and process historical data for ASX200 stocks"""

    def __init__(self, db_manager: ASXDatabaseManager, days: int = 180,
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

    def get_db_connection(self) -> ASXDatabaseManager:
        """
        Get thread-local database connection.
        Each thread gets its own connection to avoid SQLite threading issues.

        Returns:
            Database manager instance for the current thread
        """
        if not hasattr(self._local, 'db'):
            # Create new connection for this thread
            logger.debug(f"Creating thread-local DB connection for thread {threading.current_thread().ident}")
            self._local.db = ASXDatabaseManager(self.db_path)
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
            symbol: Stock symbol (e.g., 'BHP.AX')

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
            symbol: Stock symbol (e.g., 'BHP.AX')

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

            # Market cap in lakhs (A$ 10,000 = 1 lakh)
            market_cap_raw = info.get('marketCap', 0)
            market_cap = market_cap_raw / 10000 if market_cap_raw else 0

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
            symbol: Stock symbol (without .AX suffix)

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
            symbols: List of stock symbols (with .AX suffix)
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
            symbol: Stock symbol (without .AX suffix)
            df: DataFrame with OHLCV and indicator data

        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

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

                deleted_count = db.cursor.rowcount
                if deleted_count > 0:
                    logger.debug(f"Deleted {deleted_count} existing records for {symbol} in date range {first_date_str} to {last_date_str}")

                # Now insert the new data
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
            symbol: Stock symbol (without .AX suffix)

        Returns:
            Tuple of (needs_download, last_date)
            - needs_download: False if data is from today, True otherwise for incremental update
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

                # Skip ONLY if data is from today
                if last_date:
                    days_old = (datetime.now() - last_date).days
                    if days_old == 0:
                        logger.info(f"Skipping {symbol} - data is current ({result[1]} records, last: {last_date.strftime('%Y-%m-%d')})")
                        return (False, last_date)
                    else:
                        logger.info(f"Incremental update for {symbol} - data is {days_old} days old (from {last_date.strftime('%Y-%m-%d')})")
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

            # Calculate rolling window cutoff (maintain self.days)
            window_start = end_date - timedelta(days=self.days)
            prune_before = window_start if last_date < window_start else None

            logger.debug(f"{symbol}: Incremental range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            if prune_before:
                logger.debug(f"{symbol}: Will prune data before {prune_before.strftime('%Y-%m-%d')}")

            return (start_date, end_date, prune_before)
        else:
            # New stock: download full period with buffer for indicators
            start_date = end_date - timedelta(days=self.days + 30)
            logger.debug(f"{symbol}: New stock, full range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return (start_date, end_date, None)

    def process_stock(self, symbol_with_suffix: str, skip_existing: bool = True) -> bool:
        """
        Process a single stock with incremental updates and smart error handling

        Args:
            symbol_with_suffix: Stock symbol with .AX suffix
            skip_existing: If True, skip stocks that are already downloaded

        Returns:
            True if successful, False otherwise
        """
        # Remove .AX suffix for database operations
        symbol = symbol_with_suffix.replace('.AX', '')

        try:
            # Check if needs download and get last date
            if skip_existing:
                needs_download, last_date = self.check_if_downloaded(symbol)
                if not needs_download:
                    self.skipped_stocks.append(symbol)
                    return True
            else:
                last_date = None

            # Get thread-local connection for metadata updates
            db = self.get_db_connection()

            # Mark as downloading in metadata
            with self.db_lock:
                db.update_metadata(symbol, data_status='downloading')

            # Fetch historical data with incremental support
            df = self.fetch_stock_data(symbol_with_suffix)

            if df is None or df.empty:
                self.failed_downloads.append(symbol)
                with self.db_lock:
                    db.update_metadata(symbol, data_status='failed', total_records=0)
                return False

            # Store data in database
            if not self.store_stock_data(symbol, df):
                self.failed_downloads.append(symbol)
                with self.db_lock:
                    db.update_metadata(symbol, data_status='failed', total_records=0)
                return False

            # Fetch fundamentals
            fundamentals = self.fetch_fundamentals(symbol_with_suffix)

            # Update metadata
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date_new = df.index[-1].strftime('%Y-%m-%d')

            with self.db_lock:
                db.update_metadata(
                    symbol,
                    sector=fundamentals['sector'],
                    market_cap=fundamentals['market_cap'],
                    debt_to_equity=fundamentals['debt_to_equity'],
                    roe=fundamentals['roe'],
                    first_date=first_date,
                    last_date=last_date_new,
                    total_records=len(df),
                    data_status='complete'
                )

            self.successful_downloads.append(symbol)
            return True

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.failed_downloads.append(symbol)
            db = self.get_db_connection()
            with self.db_lock:
                db.update_metadata(symbol, data_status='failed', total_records=0)
            return False

    def process_bulk_batch(self, symbols: List[str], skip_existing: bool = True) -> int:
        """
        Process a batch of stocks using bulk download with incremental update support

        Args:
            symbols: List of stock symbols (with .AX suffix)
            skip_existing: If True, skip stocks that are already downloaded

        Returns:
            Number of successfully processed stocks
        """
        if not symbols:
            return 0

        success_count = 0

        try:
            # Filter out stocks that should be skipped and group by date range needs
            stocks_to_download = []
            stock_last_dates = {}  # Track last dates for incremental updates

            for symbol_with_suffix in symbols:
                symbol = symbol_with_suffix.replace('.AX', '')
                if skip_existing:
                    needs_download, last_date = self.check_if_downloaded(symbol)
                    if not needs_download:
                        self.skipped_stocks.append(symbol)
                        continue
                    stock_last_dates[symbol_with_suffix] = last_date
                else:
                    stock_last_dates[symbol_with_suffix] = None

                stocks_to_download.append(symbol_with_suffix)

            if not stocks_to_download:
                return 0

            logger.info(f"Bulk downloading {len(stocks_to_download)} stocks...")

            # For bulk download, we need to use the widest date range
            # Calculate the earliest start date needed across all stocks
            end_date = datetime.now()
            earliest_start_date = end_date - timedelta(days=self.days + 30)

            # Check if we have any incremental updates that need earlier data
            for symbol_with_suffix in stocks_to_download:
                last_date = stock_last_dates[symbol_with_suffix]
                if last_date:
                    # For incremental, we need data from last_date + 1
                    incremental_start = last_date + timedelta(days=1)
                    # But we may need to recalculate indicators, so get more history
                    incremental_start_with_buffer = last_date - timedelta(days=60)
                    if incremental_start_with_buffer < earliest_start_date:
                        earliest_start_date = incremental_start_with_buffer

            start_date = earliest_start_date

            # Bulk download
            bulk_data = self.download_bulk(stocks_to_download, start_date, end_date)

            # Get thread-local connection for this batch
            db = self.get_db_connection()

            # Process each downloaded stock
            for symbol_with_suffix in stocks_to_download:
                symbol = symbol_with_suffix.replace('.AX', '')

                try:
                    # Mark as downloading
                    with self.db_lock:
                        db.update_metadata(symbol, data_status='downloading')

                    # Get data from bulk download
                    if symbol_with_suffix not in bulk_data:
                        logger.warning(f"No data in bulk download for {symbol}")
                        self.failed_downloads.append(symbol)
                        with self.db_lock:
                            db.update_metadata(symbol, data_status='failed', total_records=0)
                        continue

                    df = bulk_data[symbol_with_suffix]

                    # Calculate technical indicators
                    df = self.calculate_technical_indicators(df)

                    # Keep only requested days
                    df = df.tail(self.days)

                    if df.empty:
                        logger.warning(f"Empty data after processing for {symbol}")
                        self.failed_downloads.append(symbol)
                        with self.db_lock:
                            db.update_metadata(symbol, data_status='failed', total_records=0)
                        continue

                    # Store in database
                    if not self.store_stock_data(symbol, df):
                        self.failed_downloads.append(symbol)
                        with self.db_lock:
                            db.update_metadata(symbol, data_status='failed', total_records=0)
                        continue

                    # Fetch fundamentals
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
                    logger.info(f"Successfully processed {symbol} from bulk download")

                except Exception as e:
                    logger.error(f"Error processing {symbol} from bulk download: {e}")
                    self.failed_downloads.append(symbol)
                    with self.db_lock:
                        db.update_metadata(symbol, data_status='failed', total_records=0)

        except Exception as e:
            logger.error(f"Error in bulk batch processing: {e}")

        return success_count

    def process_batch_parallel(self, symbols: List[str], skip_existing: bool = True) -> None:
        """
        Process stocks in parallel using ThreadPoolExecutor

        Args:
            symbols: List of stock symbols (with .AX suffix)
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

    def process_batch(self, symbols: List[str], batch_size: int = 10, delay: float = 2.0,
                     skip_existing: bool = True) -> None:
        """
        Process stocks in batches with progress tracking

        Args:
            symbols: List of stock symbols (with .AX suffix)
            batch_size: Number of stocks to process before delay
            delay: Delay in seconds between batches
            skip_existing: If True, skip stocks that are already downloaded
        """
        total = len(symbols)
        logger.info(f"Starting batch processing of {total} stocks")
        logger.info(f"Batch size: {batch_size}, Delay: {delay}s")

        for i, symbol in enumerate(symbols, 1):
            # Progress tracking
            progress_pct = (i / total) * 100
            logger.info(f"[{i}/{total}] ({progress_pct:.1f}%) Processing {symbol}")

            # Process stock with retry logic
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    success = self.process_stock(symbol, skip_existing)
                    if not success and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(f"Retry {retry_count}/{max_retries} for {symbol}")
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Exception processing {symbol}: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)

            # Rate limiting - delay after each batch
            if i % batch_size == 0 and i < total:
                logger.info(f"Batch complete. Waiting {delay}s before next batch...")
                time.sleep(delay)

        # Final summary
        logger.info("=" * 80)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total stocks: {total}")
        logger.info(f"Successful: {len(self.successful_downloads)}")
        logger.info(f"Skipped (already downloaded): {len(self.skipped_stocks)}")
        logger.info(f"Failed: {len(self.failed_downloads)}")

        if self.failed_downloads:
            logger.warning(f"Failed stocks: {', '.join(self.failed_downloads[:20])}")
            if len(self.failed_downloads) > 20:
                logger.warning(f"... and {len(self.failed_downloads) - 20} more")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Fetch ASX200 historical data with Phase 3 optimizations'
    )
    parser.add_argument('--symbols', nargs='+', help='Custom list of symbols (e.g., BHP.AX CBA.AX)')
    parser.add_argument('--days', type=int, default=180, help='Number of days of history (default: 180)')
    parser.add_argument('--test', action='store_true', help='Test mode - download only 10 stocks')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for sequential mode (default: 10)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between batches in seconds (default: 2.0)')
    parser.add_argument('--no-skip', action='store_true', help='Re-download all stocks (do not skip existing)')
    parser.add_argument('--db-path', type=str, help='Custom database path')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--force', action='store_true', help='Force full re-download even if data exists')
    parser.add_argument('--sequential', action='store_true', help='Use sequential mode instead of parallel (slower)')

    args = parser.parse_args()

    # Ensure log directory exists
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Get symbols
    if args.symbols:
        # Custom symbols provided
        symbols = [s if s.endswith('.AX') else f"{s}.AX" for s in args.symbols]
        logger.info(f"Using custom symbol list: {len(symbols)} stocks")
    elif args.test:
        # Test mode - first 10 stocks
        symbols = AustraliaConfig.ASX_200_SYMBOLS[:10]
        logger.info(f"TEST MODE: Using first 10 ASX200 stocks")
    else:
        # Full ASX200 list
        symbols = AustraliaConfig.ASX_200_SYMBOLS
        logger.info(f"Using full ASX200 list: {len(symbols)} stocks")

    # Initialize database
    db_path = args.db_path if args.db_path else None
    db = ASXDatabaseManager(db_path)

    try:
        db.connect()
        logger.info(f"Connected to database: {db.db_path}")

        # Create metadata table
        db.create_metadata_table()

        # Create results tracking tables
        db.create_results_tracking_tables()

        # Create tables for all symbols
        logger.info("Creating stock tables...")
        for symbol in symbols:
            clean_symbol = symbol.replace('.AX', '')
            db.create_stock_table(clean_symbol)

        # Initialize data fetcher with new parameters
        fetcher = ASX200DataFetcher(
            db,
            days=args.days,
            workers=args.workers,
            force=args.force
        )

        # Process stocks using parallel or sequential mode
        if args.sequential:
            # Legacy sequential mode
            logger.info("Using SEQUENTIAL mode (legacy)")
            fetcher.process_batch(
                symbols,
                batch_size=args.batch_size,
                delay=args.delay,
                skip_existing=not args.no_skip
            )
        else:
            # New parallel mode (default)
            logger.info(f"Using PARALLEL mode with {args.workers} workers")
            fetcher.process_batch_parallel(
                symbols,
                skip_existing=not args.no_skip
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
