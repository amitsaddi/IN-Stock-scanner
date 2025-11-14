"""
Database schema for ASX200 historical data storage
Each stock gets its own table with common structure
"""
import sqlite3
import logging
from typing import List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASXDatabaseManager:
    """Manages SQLite database for ASX200 historical data"""

    def __init__(self, db_path: str = None):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "asx200_historical.db"

        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def create_stock_table(self, symbol: str):
        """
        Create table for individual stock

        Table structure:
        - date: Trading date (PRIMARY KEY)
        - open, high, low, close: OHLC prices
        - volume: Trading volume
        - adj_close: Adjusted close price
        - ema_20, ema_50, sma_200: Moving averages
        - rsi: Relative Strength Index
        - macd, macd_signal, macd_hist: MACD indicators
        - atr: Average True Range
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - volume_20d_avg: 20-day volume average
        - volume_ratio: Current volume / 20-day average
        - week_52_high, week_52_low: 52-week high/low
        - week_52_high_proximity: Distance to 52-week high (%)

        Args:
            symbol: Stock symbol (e.g., 'BHP' without .AX)
        """
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            -- Date and Price Data
            date TEXT PRIMARY KEY,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            adj_close REAL,

            -- Moving Averages
            ema_20 REAL,
            ema_50 REAL,
            sma_200 REAL,

            -- Momentum Indicators
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,

            -- Volatility Indicators
            atr REAL,
            atr_percent REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,

            -- Volume Indicators
            volume_20d_avg REAL,
            volume_ratio REAL,

            -- 52-Week Metrics
            week_52_high REAL,
            week_52_low REAL,
            week_52_high_proximity REAL,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            self.cursor.execute(create_table_sql)
            self.conn.commit()
            logger.info(f"Created table: {table_name}")

            # Create indexes
            self._create_indexes(table_name)

        except sqlite3.Error as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    def _create_indexes(self, table_name: str):
        """
        Create indexes for efficient querying

        Args:
            table_name: Name of the stock table
        """
        indexes = [
            # Date index (already primary key, but explicit)
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date DESC);",

            # RSI index for filtering
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_rsi ON {table_name}(rsi);",

            # Volume ratio index
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_volume_ratio ON {table_name}(volume_ratio);",

            # 52-week high proximity index
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_52w_prox ON {table_name}(week_52_high_proximity);",

            # Composite index for common queries
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_scan ON {table_name}(date, rsi, volume_ratio, week_52_high_proximity);",
        ]

        for idx_sql in indexes:
            try:
                self.cursor.execute(idx_sql)
            except sqlite3.Error as e:
                logger.warning(f"Error creating index: {e}")

        self.conn.commit()
        logger.info(f"Created indexes for {table_name}")

    def create_metadata_table(self):
        """
        Create metadata table to track stocks and data status
        """
        create_metadata_sql = """
        CREATE TABLE IF NOT EXISTS stock_metadata (
            symbol TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            sector TEXT,
            market_cap REAL,
            debt_to_equity REAL,
            roe REAL,
            first_date TEXT,
            last_date TEXT,
            total_records INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_status TEXT DEFAULT 'pending'
        );
        """

        try:
            self.cursor.execute(create_metadata_sql)

            # Create index on symbol
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_symbol
                ON stock_metadata(symbol);
            """)

            # Create index on sector for sector-based queries
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_sector
                ON stock_metadata(sector);
            """)

            self.conn.commit()
            logger.info("Created stock_metadata table")

        except sqlite3.Error as e:
            logger.error(f"Error creating metadata table: {e}")
            raise

    def create_results_tracking_tables(self):
        """
        Create tables for tracking backtest results history
        """
        try:
            # Results History Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE NOT NULL,
                    strategy_type TEXT NOT NULL DEFAULT 'swing',
                    version TEXT NOT NULL CHECK(version IN ('v1', 'v2')),

                    -- Core Metrics
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    avg_return REAL,
                    total_return REAL,

                    -- Risk Metrics
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    avg_hold_days REAL,
                    median_hold_days REAL,
                    return_volatility REAL,

                    -- Backtest Configuration
                    backtest_start_date DATE,
                    backtest_end_date DATE,
                    total_stocks_scanned INTEGER,

                    -- Additional Data (JSON)
                    sector_performance TEXT,
                    signal_distribution TEXT,

                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(run_date, strategy_type, version)
                )
            """)

            # Comparison Results Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparison_results_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE NOT NULL,
                    strategy_type TEXT NOT NULL DEFAULT 'swing',

                    -- Recommendation
                    recommendation TEXT CHECK(recommendation IN ('use_v1', 'use_v2', 'inconclusive')),
                    confidence_level TEXT CHECK(confidence_level IN ('high', 'medium', 'low')),
                    criteria_met INTEGER,
                    total_criteria INTEGER,

                    -- Statistical Tests
                    returns_ttest_pvalue REAL,
                    winrate_chisquare_pvalue REAL,

                    -- Performance Delta (v2 - v1)
                    delta_total_trades INTEGER,
                    delta_win_rate REAL,
                    delta_avg_return REAL,
                    delta_sharpe_ratio REAL,
                    delta_max_drawdown REAL,

                    -- Report Files
                    html_report_path TEXT,
                    summary_report_path TEXT,
                    csv_report_path TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(run_date, strategy_type)
                )
            """)

            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_date ON backtest_results_history(run_date)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_strategy ON backtest_results_history(strategy_type, version)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_comparison_date ON comparison_results_history(run_date)")

            # Create view for 30-day trends
            self.cursor.execute("""
                CREATE VIEW IF NOT EXISTS vw_30day_trend AS
                SELECT
                    run_date,
                    strategy_type,
                    recommendation,
                    confidence_level,
                    criteria_met,
                    total_criteria,
                    delta_avg_return,
                    delta_sharpe_ratio,
                    CASE
                        WHEN recommendation = 'use_v2' THEN 1
                        WHEN recommendation = 'use_v1' THEN -1
                        ELSE 0
                    END as score
                FROM comparison_results_history
                WHERE run_date >= date('now', '-30 days')
                ORDER BY run_date DESC
            """)

            self.conn.commit()
            logger.info("Created backtest results tracking tables and views")

        except sqlite3.Error as e:
            logger.error(f"Error creating results tracking tables: {e}")
            raise

    def update_metadata(self, symbol: str, **kwargs):
        """
        Update metadata for a stock

        Args:
            symbol: Stock symbol
            **kwargs: Metadata fields to update
        """
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

        # Check if record exists
        self.cursor.execute(
            "SELECT symbol FROM stock_metadata WHERE symbol = ?",
            (symbol,)
        )
        exists = self.cursor.fetchone()

        if exists:
            # Update existing record
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            set_clause += ", last_updated = CURRENT_TIMESTAMP"
            values = list(kwargs.values()) + [symbol]

            update_sql = f"""
                UPDATE stock_metadata
                SET {set_clause}
                WHERE symbol = ?
            """
            self.cursor.execute(update_sql, values)
        else:
            # Insert new record
            kwargs['symbol'] = symbol
            kwargs['table_name'] = table_name

            columns = ", ".join(kwargs.keys())
            placeholders = ", ".join(["?" for _ in kwargs])

            insert_sql = f"""
                INSERT INTO stock_metadata ({columns})
                VALUES ({placeholders})
            """
            self.cursor.execute(insert_sql, list(kwargs.values()))

        self.conn.commit()

    def create_unified_view(self):
        """
        Create unified view across all stock tables
        This allows querying all stocks together
        """
        # First, get all stock tables
        self.cursor.execute("""
            SELECT table_name FROM stock_metadata
        """)

        stock_tables = [row[0] for row in self.cursor.fetchall()]

        if not stock_tables:
            logger.warning("No stock tables found. Cannot create unified view.")
            return

        # Build UNION query
        union_queries = []
        for table_name in stock_tables:
            # Extract symbol from table name
            symbol = table_name.replace('stock_', '').replace('_', '.')

            query = f"""
                SELECT
                    '{symbol}' as symbol,
                    date, open, high, low, close, volume, adj_close,
                    ema_20, ema_50, sma_200,
                    rsi, macd, macd_signal, macd_hist,
                    atr, atr_percent,
                    bb_upper, bb_middle, bb_lower,
                    volume_20d_avg, volume_ratio,
                    week_52_high, week_52_low, week_52_high_proximity
                FROM {table_name}
            """
            union_queries.append(query)

        # Create view
        view_sql = f"""
        CREATE VIEW IF NOT EXISTS vw_all_stocks AS
        {' UNION ALL '.join(union_queries)}
        """

        try:
            # Drop existing view first
            self.cursor.execute("DROP VIEW IF EXISTS vw_all_stocks")

            self.cursor.execute(view_sql)
            self.conn.commit()
            logger.info("Created unified view: vw_all_stocks")

        except sqlite3.Error as e:
            logger.error(f"Error creating unified view: {e}")
            raise

    def get_stock_count(self) -> int:
        """Get total number of stocks in database"""
        self.cursor.execute("SELECT COUNT(*) FROM stock_metadata")
        return self.cursor.fetchone()[0]

    def get_stock_list(self) -> List[str]:
        """Get list of all stocks in database"""
        self.cursor.execute("SELECT symbol FROM stock_metadata ORDER BY symbol")
        return [row[0] for row in self.cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple:
        """
        Get date range for a stock

        Returns:
            (first_date, last_date)
        """
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

        self.cursor.execute(f"""
            SELECT MIN(date), MAX(date)
            FROM {table_name}
        """)

        return self.cursor.fetchone()

    def prune_old_data(self, symbol: str, before_date: str):
        """
        Remove data older than specified date to maintain rolling window

        Args:
            symbol: Stock symbol
            before_date: Delete records before this date (YYYY-MM-DD)
        """
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_').replace('&', '_')}"
        try:
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE date < ?
            """, (before_date,))
            self.conn.commit()
            deleted = self.cursor.rowcount
            if deleted > 0:
                logger.info(f"Pruned {deleted} old records from {table_name} (before {before_date})")
        except Exception as e:
            logger.error(f"Error pruning data for {symbol}: {e}")

    def vacuum(self):
        """Optimize database (reclaim space, rebuild indexes)"""
        logger.info("Running VACUUM to optimize database...")
        self.cursor.execute("VACUUM")
        logger.info("Database optimized")


def initialize_database(symbols: List[str], db_path: str = None):
    """
    Initialize database with tables for all symbols

    Args:
        symbols: List of stock symbols
        db_path: Path to database file
    """
    db = ASXDatabaseManager(db_path)
    db.connect()

    try:
        # Create metadata table
        db.create_metadata_table()

        # Create table for each stock
        for symbol in symbols:
            clean_symbol = symbol.replace('.AX', '')
            db.create_stock_table(clean_symbol)

            # Add to metadata
            db.update_metadata(
                clean_symbol,
                data_status='pending'
            )

        logger.info(f"Initialized database with {len(symbols)} stock tables")

        # Note: Unified view will be created after data is loaded

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Test with a few stocks
    test_symbols = ["BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX"]

    print("Initializing test database...")
    initialize_database(test_symbols, db_path="AU-testing/data/test.db")
    print("Done!")

    # Test connection and queries
    db = ASXDatabaseManager(db_path="AU-testing/data/test.db")
    db.connect()

    print(f"\nTotal stocks: {db.get_stock_count()}")
    print(f"Stock list: {db.get_stock_list()}")

    db.close()
