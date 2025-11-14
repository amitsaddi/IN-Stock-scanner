"""
Comprehensive Backtesting Engine for v1 Swing Trading Strategy (ASX200)

This module implements a complete backtesting framework for the current production
swing trading strategy, simulating daily scans over historical data and tracking
trade performance metrics.

Strategy v1 Criteria:
- Fundamental: Market Cap >= A$500M, D/E <= 1.0, ROE >= 10%
- Technical: RSI 35-65, 52W High 85-98%, Volume >= 1.2x, Price > EMA(20) preferred
- Entry Signals: Pullback, Breakout, MACD Cross, MA Cross, Trend Follow
- Risk Management: 3-15 day hold, 12-15% targets, 5-7% stop loss

Author: Backtesting Framework
Date: 2025-11-11
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import logging
import json
import argparse
from pathlib import Path
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Trade:
    """Represents a single trade with entry/exit details"""
    trade_id: int
    symbol: str
    sector: str
    entry_date: str
    entry_price: float
    entry_signal: str
    score: float

    # Exit details (filled when trade closes)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'target', 'stop_loss', 'max_hold'

    # Trade metrics
    hold_days: Optional[int] = None
    profit_loss_pct: Optional[float] = None
    profit_loss_abs: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

    # Additional metrics
    rsi_entry: Optional[float] = None
    volume_ratio_entry: Optional[float] = None
    week_52_high_proximity: Optional[float] = None

    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_date is None

    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.profit_loss_pct is not None and self.profit_loss_pct > 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results"""
    # Date range
    start_date: str
    end_date: str
    total_days: int
    trading_days: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Returns
    avg_gain_pct: float
    avg_loss_pct: float
    avg_trade_pct: float
    total_return_pct: float

    # Risk metrics
    profit_factor: float  # Gross profit / Gross loss
    max_drawdown_pct: float
    sharpe_ratio: float

    # Hold period
    avg_hold_days: float
    median_hold_days: float

    # Activity metrics
    signals_per_week: float
    avg_candidates_per_scan: float

    # Entry type breakdown
    entry_type_performance: Dict[str, Dict]

    # Sector breakdown
    sector_performance: Dict[str, Dict]

    # Distribution
    return_distribution: Dict[str, int]  # Bins of returns


# ============================================================================
# SECTOR WEIGHTS (V1)
# ============================================================================

V1_SECTOR_WEIGHTS = {
    'Materials': 1.2,
    'Energy': 1.15,
    'Financials': 1.0,
    'Consumer Discretionary': 1.0,
    'Industrials': 1.0,
    'Health Care': 0.9,
    'Information Technology': 0.8,
    'Communication Services': 0.85,
    'Consumer Staples': 0.95,
    'Utilities': 0.9,
    'Real Estate': 0.85
}


# ============================================================================
# ENTRY SIGNAL DEFINITIONS (V1)
# ============================================================================

ENTRY_SIGNAL_PARAMS = {
    'pullback': {'target_pct': 12, 'stop_loss_pct': 7},
    'breakout': {'target_pct': 15, 'stop_loss_pct': 5},
    'macd_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'ma_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'trend_follow': {'target_pct': 15, 'stop_loss_pct': 6}
}


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestV1:
    """
    Comprehensive backtesting engine for v1 swing trading strategy
    """

    def __init__(self, db_path: str, verbose: bool = False):
        """
        Initialize backtesting engine

        Args:
            db_path: Path to SQLite database with historical data
            verbose: Enable detailed logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self.trades: List[Trade] = []
        self.trade_counter = 0

        if verbose:
            logger.setLevel(logging.DEBUG)

    def connect_db(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def get_stock_metadata(self) -> pd.DataFrame:
        """
        Get stock metadata including fundamentals

        Returns:
            DataFrame with symbol, sector, fundamentals
        """
        query = """
        SELECT
            symbol,
            sector,
            market_cap,
            debt_to_equity,
            roe,
            table_name
        FROM stock_metadata
        WHERE data_status = 'complete'
        """

        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded metadata for {len(df)} stocks")
        return df

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical price and indicator data for a stock

        Args:
            symbol: Stock symbol (without .AX)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV and technical indicators
        """
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

        query = f"""
        SELECT
            date, open, high, low, close, volume,
            ema_20, ema_50, sma_200,
            rsi, macd, macd_signal, macd_hist,
            volume_ratio, week_52_high, week_52_high_proximity
        FROM {table_name}
        WHERE date >= ? AND date <= ?
        ORDER BY date ASC
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except sqlite3.Error as e:
            if self.verbose:
                logger.debug(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def apply_fundamental_filters(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Apply v1 fundamental filters

        Criteria:
        - Market Cap >= A$500M (50,000 lakhs)
        - Debt-to-Equity <= 1.0
        - ROE >= 10%

        Args:
            metadata: Stock metadata DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered = metadata[
            (metadata['market_cap'] >= 50000) &  # A$500M = 50,000 lakhs
            (metadata['debt_to_equity'] <= 1.0) &
            (metadata['roe'] >= 10.0)
        ].copy()

        logger.debug(f"Fundamental filters: {len(metadata)} -> {len(filtered)} stocks")
        return filtered

    def apply_technical_filters(self, df: pd.DataFrame) -> bool:
        """
        Apply v1 technical filters to latest data point

        Criteria:
        - RSI: 35-65
        - 52-week high proximity: 85-98%
        - Volume ratio >= 1.2x

        Args:
            df: Historical data with latest point to check

        Returns:
            True if passes all technical filters
        """
        if df.empty or len(df) < 1:
            return False

        latest = df.iloc[-1]

        # RSI check
        rsi = latest.get('rsi', 0)
        if not (35 <= rsi <= 65):
            return False

        # 52-week high proximity check
        week_52_prox = latest.get('week_52_high_proximity', 0)
        if not (85 <= week_52_prox <= 98):
            return False

        # Volume ratio check
        volume_ratio = latest.get('volume_ratio', 0)
        if volume_ratio < 1.2:
            return False

        return True

    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v1 technical score (35% of total)

        Components:
        - RSI positioning (optimal at 50)
        - MACD signals
        - MA alignment

        Args:
            df: Historical data

        Returns:
            Technical score (0-35)
        """
        if df.empty or len(df) < 2:
            return 0

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0

        # RSI positioning (0-12 points) - optimal at 50
        rsi = latest.get('rsi', 50)
        rsi_score = 12 * (1 - abs(rsi - 50) / 50)
        score += max(0, rsi_score)

        # MACD signals (0-12 points)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_hist', 0)

        if macd > macd_signal and macd_hist > 0:
            score += 12
        elif macd > macd_signal:
            score += 6

        # MA alignment (0-11 points)
        close = latest['close']
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)

        if close > ema_20 > ema_50 > sma_200:
            score += 11
        elif close > ema_20 > ema_50:
            score += 8
        elif close > ema_20:
            score += 5

        return min(35, score)

    def calculate_fundamental_score(self, market_cap: float, roe: float, debt_equity: float) -> float:
        """
        Calculate v1 fundamental score (25% of total)

        Components:
        - ROE (higher is better)
        - Debt/Equity (lower is better)
        - Market Cap (larger is better)

        Args:
            market_cap: Market cap in lakhs
            roe: Return on equity %
            debt_equity: Debt-to-equity ratio

        Returns:
            Fundamental score (0-25)
        """
        score = 0

        # ROE score (0-10 points) - normalize around 10-30%
        roe_normalized = min((roe - 10) / 20, 1.0)
        score += 10 * max(0, roe_normalized)

        # Debt/Equity score (0-8 points) - lower is better
        de_score = (1.0 - debt_equity) * 8
        score += max(0, de_score)

        # Market cap score (0-7 points) - logarithmic scale
        # 50,000 lakhs (A$500M) = baseline, 500,000+ lakhs = max
        if market_cap >= 50000:
            mcap_normalized = min(np.log10(market_cap / 50000) / np.log10(10), 1.0)
            score += 7 * mcap_normalized

        return min(25, score)

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v1 momentum score (25% of total)

        Primary component:
        - 52-week high proximity

        Args:
            df: Historical data

        Returns:
            Momentum score (0-25)
        """
        if df.empty:
            return 0

        latest = df.iloc[-1]
        week_52_prox = latest.get('week_52_high_proximity', 0)

        # Linear scoring: 85% = 0 points, 98% = 25 points
        score = ((week_52_prox - 85) / 13) * 25
        return max(0, min(25, score))

    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v1 volume score (15% of total)

        Component:
        - Volume ratio vs 20-day average

        Args:
            df: Historical data

        Returns:
            Volume score (0-15)
        """
        if df.empty:
            return 0

        latest = df.iloc[-1]
        volume_ratio = latest.get('volume_ratio', 0)

        # 1.2x = 0 points, 2.0x+ = 15 points
        score = ((volume_ratio - 1.2) / 0.8) * 15
        return max(0, min(15, score))

    def calculate_total_score(self, df: pd.DataFrame, market_cap: float,
                             roe: float, debt_equity: float, sector: str) -> float:
        """
        Calculate total v1 score with sector weighting

        Args:
            df: Historical data
            market_cap: Market cap in lakhs
            roe: Return on equity
            debt_equity: Debt-to-equity ratio
            sector: Stock sector

        Returns:
            Total score (0-100+, sector weighted)
        """
        tech_score = self.calculate_technical_score(df)
        fund_score = self.calculate_fundamental_score(market_cap, roe, debt_equity)
        momentum_score = self.calculate_momentum_score(df)
        volume_score = self.calculate_volume_score(df)

        base_score = tech_score + fund_score + momentum_score + volume_score

        # Apply sector weight
        sector_weight = V1_SECTOR_WEIGHTS.get(sector, 1.0)
        final_score = base_score * sector_weight

        return final_score

    def identify_entry_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify entry signal type based on v1 criteria

        Entry Signals:
        1. Pullback: RSI 30-35, near support
        2. Breakout: Price > EMA_20 with volume
        3. MACD Cross: Fresh bullish crossover
        4. MA Cross: EMA(20) crosses above EMA(50)
        5. Trend Follow: Above all MAs, near 52W high

        Args:
            df: Historical data (need at least 2 days)

        Returns:
            Entry signal type or None
        """
        if df.empty or len(df) < 2:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest.get('rsi', 50)
        close = latest['close']
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)
        volume_ratio = latest.get('volume_ratio', 1.0)
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = prev.get('macd_hist', 0)
        week_52_prox = latest.get('week_52_high_proximity', 0)

        # Priority order for signal detection

        # 1. Pullback (highest priority for mean reversion)
        if 30 <= rsi <= 35 and close < ema_20:
            return 'pullback'

        # 2. MACD Cross (fresh bullish crossover)
        if macd_hist > 0 and prev_macd_hist <= 0:
            return 'macd_cross'

        # 3. MA Cross (EMA 20 just crossed EMA 50)
        prev_ema_20 = prev.get('ema_20', 0)
        prev_ema_50 = prev.get('ema_50', 0)
        if ema_20 > ema_50 and prev_ema_20 <= prev_ema_50:
            return 'ma_cross'

        # 4. Breakout (price breaks above EMA 20 with volume)
        if close > ema_20 * 1.02 and volume_ratio > 1.5:
            return 'breakout'

        # 5. Trend Follow (above all MAs, near 52W high)
        if close > ema_20 > ema_50 > sma_200 and week_52_prox >= 90:
            return 'trend_follow'

        return None

    def calculate_targets(self, entry_price: float, signal_type: str) -> Tuple[float, float]:
        """
        Calculate target and stop loss based on entry signal

        Args:
            entry_price: Entry price
            signal_type: Entry signal type

        Returns:
            (target_price, stop_loss_price)
        """
        params = ENTRY_SIGNAL_PARAMS.get(signal_type, {'target_pct': 15, 'stop_loss_pct': 6})

        target = entry_price * (1 + params['target_pct'] / 100)
        stop_loss = entry_price * (1 - params['stop_loss_pct'] / 100)

        return round(target, 2), round(stop_loss, 2)

    def scan_date(self, scan_date: str, metadata: pd.DataFrame) -> List[Dict]:
        """
        Simulate a single day's scan

        Args:
            scan_date: Date to scan (YYYY-MM-DD)
            metadata: Stock metadata with fundamentals

        Returns:
            List of candidate dictionaries with scores
        """
        candidates = []

        # Apply fundamental filters first
        filtered_metadata = self.apply_fundamental_filters(metadata)

        for _, stock in filtered_metadata.iterrows():
            symbol = stock['symbol']
            sector = stock['sector']
            market_cap = stock['market_cap']
            roe = stock['roe']
            debt_equity = stock['debt_to_equity']

            # Get historical data up to scan date (need lookback for indicators)
            lookback_date = (pd.to_datetime(scan_date) - timedelta(days=100)).strftime('%Y-%m-%d')
            df = self.get_historical_data(symbol, lookback_date, scan_date)

            if df.empty or len(df) < 50:
                continue

            # Apply technical filters
            if not self.apply_technical_filters(df):
                continue

            # Calculate score
            score = self.calculate_total_score(df, market_cap, roe, debt_equity, sector)

            # Check minimum score threshold (60 for v1)
            if score < 60:
                continue

            # Identify entry signal
            entry_signal = self.identify_entry_signal(df)
            if entry_signal is None:
                continue

            latest = df.iloc[-1]

            candidates.append({
                'symbol': symbol,
                'sector': sector,
                'score': score,
                'entry_signal': entry_signal,
                'entry_price': latest['close'],
                'rsi': latest.get('rsi', 0),
                'volume_ratio': latest.get('volume_ratio', 0),
                'week_52_high_proximity': latest.get('week_52_high_proximity', 0)
            })

        # Sort by score and take top 15 (v1 max results)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:15]

    def check_exit_conditions(self, trade: Trade, current_date: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if trade should be exited

        Conditions:
        1. Target hit
        2. Stop loss hit
        3. Max hold period (15 days)

        Args:
            trade: Open trade
            current_date: Current date to check

        Returns:
            (should_exit, exit_reason, exit_price)
        """
        # Get price data for current date
        df = self.get_historical_data(trade.symbol, current_date, current_date)

        if df.empty:
            return False, None, None

        current_data = df.iloc[0]
        high = current_data['high']
        low = current_data['low']
        close = current_data['close']

        # Check target hit (use intraday high)
        if high >= trade.target_price:
            return True, 'target', trade.target_price

        # Check stop loss hit (use intraday low)
        if low <= trade.stop_loss_price:
            return True, 'stop_loss', trade.stop_loss_price

        # Check max hold period
        entry_date = pd.to_datetime(trade.entry_date)
        current_dt = pd.to_datetime(current_date)
        hold_days = (current_dt - entry_date).days

        if hold_days >= 15:
            return True, 'max_hold', close

        return False, None, None

    def run_backtest(self, start_date: str, end_date: str) -> List[Trade]:
        """
        Run complete backtest simulation

        Simulates daily scans, tracks trades, and calculates performance

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            List of all trades executed
        """
        logger.info("=" * 80)
        logger.info(f"Starting v1 Strategy Backtest: {start_date} to {end_date}")
        logger.info("=" * 80)

        self.connect_db()

        # Get stock metadata
        metadata = self.get_stock_metadata()

        # Generate list of trading dates
        date_range = pd.bdate_range(start=start_date, end=end_date)
        total_dates = len(date_range)

        logger.info(f"Total trading days to simulate: {total_dates}")
        logger.info(f"Stock universe: {len(metadata)} stocks")

        open_trades: List[Trade] = []
        closed_trades: List[Trade] = []

        scan_results_count = []

        for i, scan_date in enumerate(date_range):
            scan_date_str = scan_date.strftime('%Y-%m-%d')

            # Progress logging
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Progress: {i+1}/{total_dates} days | Open trades: {len(open_trades)}")

            # Step 1: Check exit conditions for all open trades
            trades_to_close = []
            for trade in open_trades:
                should_exit, exit_reason, exit_price = self.check_exit_conditions(trade, scan_date_str)

                if should_exit:
                    # Close the trade
                    trade.exit_date = scan_date_str
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason

                    entry_dt = pd.to_datetime(trade.entry_date)
                    exit_dt = pd.to_datetime(trade.exit_date)
                    trade.hold_days = (exit_dt - entry_dt).days

                    trade.profit_loss_abs = exit_price - trade.entry_price
                    trade.profit_loss_pct = ((exit_price / trade.entry_price) - 1) * 100

                    trades_to_close.append(trade)
                    closed_trades.append(trade)

                    if self.verbose:
                        logger.debug(f"CLOSE: {trade.symbol} | {exit_reason.upper()} | "
                                   f"P/L: {trade.profit_loss_pct:+.2f}% | Hold: {trade.hold_days}d")

            # Remove closed trades from open list
            for trade in trades_to_close:
                open_trades.remove(trade)

            # Step 2: Run daily scan for new candidates
            candidates = self.scan_date(scan_date_str, metadata)
            scan_results_count.append(len(candidates))

            if self.verbose and candidates:
                logger.debug(f"Scan {scan_date_str}: {len(candidates)} candidates found")

            # Step 3: Enter new positions (avoid duplicate symbols)
            open_symbols = {t.symbol for t in open_trades}

            for candidate in candidates:
                # Skip if already have open position
                if candidate['symbol'] in open_symbols:
                    continue

                # Create new trade
                entry_price = candidate['entry_price']
                signal_type = candidate['entry_signal']
                target, stop_loss = self.calculate_targets(entry_price, signal_type)

                self.trade_counter += 1
                trade = Trade(
                    trade_id=self.trade_counter,
                    symbol=candidate['symbol'],
                    sector=candidate['sector'],
                    entry_date=scan_date_str,
                    entry_price=entry_price,
                    entry_signal=signal_type,
                    score=candidate['score'],
                    target_price=target,
                    stop_loss_price=stop_loss,
                    rsi_entry=candidate['rsi'],
                    volume_ratio_entry=candidate['volume_ratio'],
                    week_52_high_proximity=candidate['week_52_high_proximity']
                )

                open_trades.append(trade)
                open_symbols.add(candidate['symbol'])

                if self.verbose:
                    logger.debug(f"OPEN: {trade.symbol} | {signal_type.upper()} | "
                               f"Entry: ${entry_price:.2f} | Target: ${target:.2f} | SL: ${stop_loss:.2f}")

        # Close any remaining open trades at end date
        for trade in open_trades:
            df = self.get_historical_data(trade.symbol, end_date, end_date)
            if not df.empty:
                exit_price = df.iloc[-1]['close']
                trade.exit_date = end_date
                trade.exit_price = exit_price
                trade.exit_reason = 'backtest_end'

                entry_dt = pd.to_datetime(trade.entry_date)
                exit_dt = pd.to_datetime(trade.exit_date)
                trade.hold_days = (exit_dt - entry_dt).days

                trade.profit_loss_abs = exit_price - trade.entry_price
                trade.profit_loss_pct = ((exit_price / trade.entry_price) - 1) * 100

                closed_trades.append(trade)

        self.trades = closed_trades

        logger.info("=" * 80)
        logger.info(f"Backtest Complete")
        logger.info(f"Total trades: {len(closed_trades)}")
        logger.info(f"Average candidates per scan: {np.mean(scan_results_count):.1f}")
        logger.info("=" * 80)

        self.close_db()
        return closed_trades

    def calculate_performance_metrics(self, trades: List[Trade],
                                     start_date: str, end_date: str) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            trades: List of completed trades
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            PerformanceMetrics object
        """
        if not trades:
            logger.warning("No trades to analyze")
            return None

        # Date calculations
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days
        trading_days = len(pd.bdate_range(start=start_date, end=end_date))

        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner())
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Returns
        gains = [t.profit_loss_pct for t in trades if t.profit_loss_pct > 0]
        losses = [t.profit_loss_pct for t in trades if t.profit_loss_pct < 0]

        avg_gain_pct = np.mean(gains) if gains else 0
        avg_loss_pct = np.mean(losses) if losses else 0
        avg_trade_pct = np.mean([t.profit_loss_pct for t in trades])
        total_return_pct = sum(t.profit_loss_pct for t in trades)

        # Risk metrics
        gross_profit = sum(gains) if gains else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Max drawdown calculation
        cumulative_returns = np.cumsum([t.profit_loss_pct for t in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown_pct = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Sharpe ratio (assuming 252 trading days per year, 0% risk-free rate)
        returns = [t.profit_loss_pct / 100 for t in trades]
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Hold period
        hold_days = [t.hold_days for t in trades if t.hold_days is not None]
        avg_hold_days = np.mean(hold_days) if hold_days else 0
        median_hold_days = np.median(hold_days) if hold_days else 0

        # Activity metrics
        weeks = total_days / 7
        signals_per_week = total_trades / weeks if weeks > 0 else 0
        avg_candidates_per_scan = total_trades / trading_days if trading_days > 0 else 0

        # Entry type breakdown
        entry_type_performance = {}
        for signal_type in ENTRY_SIGNAL_PARAMS.keys():
            signal_trades = [t for t in trades if t.entry_signal == signal_type]
            if signal_trades:
                signal_wins = sum(1 for t in signal_trades if t.is_winner())
                entry_type_performance[signal_type] = {
                    'count': len(signal_trades),
                    'win_rate': (signal_wins / len(signal_trades) * 100),
                    'avg_return': np.mean([t.profit_loss_pct for t in signal_trades])
                }

        # Sector breakdown
        sector_performance = {}
        sectors = set(t.sector for t in trades)
        for sector in sectors:
            sector_trades = [t for t in trades if t.sector == sector]
            sector_wins = sum(1 for t in sector_trades if t.is_winner())
            sector_performance[sector] = {
                'count': len(sector_trades),
                'win_rate': (sector_wins / len(sector_trades) * 100),
                'avg_return': np.mean([t.profit_loss_pct for t in sector_trades])
            }

        # Return distribution
        return_distribution = {
            '< -10%': sum(1 for t in trades if t.profit_loss_pct < -10),
            '-10% to -5%': sum(1 for t in trades if -10 <= t.profit_loss_pct < -5),
            '-5% to 0%': sum(1 for t in trades if -5 <= t.profit_loss_pct < 0),
            '0% to 5%': sum(1 for t in trades if 0 <= t.profit_loss_pct < 5),
            '5% to 10%': sum(1 for t in trades if 5 <= t.profit_loss_pct < 10),
            '10% to 15%': sum(1 for t in trades if 10 <= t.profit_loss_pct < 15),
            '> 15%': sum(1 for t in trades if t.profit_loss_pct >= 15)
        }

        metrics = PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            trading_days=trading_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_gain_pct=avg_gain_pct,
            avg_loss_pct=avg_loss_pct,
            avg_trade_pct=avg_trade_pct,
            total_return_pct=total_return_pct,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            avg_hold_days=avg_hold_days,
            median_hold_days=median_hold_days,
            signals_per_week=signals_per_week,
            avg_candidates_per_scan=avg_candidates_per_scan,
            entry_type_performance=entry_type_performance,
            sector_performance=sector_performance,
            return_distribution=return_distribution
        )

        return metrics

    def save_results(self, trades: List[Trade], metrics: PerformanceMetrics,
                    output_dir: str):
        """
        Save backtest results to files

        Generates:
        1. trades_detail.csv - All trade details
        2. performance_metrics.json - Summary metrics
        3. backtest_report.txt - Human-readable report

        Args:
            trades: List of completed trades
            metrics: Performance metrics
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Save detailed trades CSV
        trades_file = output_path / f'backtest_v1_trades_{timestamp}.csv'
        trades_df = pd.DataFrame([asdict(t) for t in trades])
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved detailed trades: {trades_file}")

        # 2. Save performance metrics JSON
        metrics_file = output_path / f'backtest_v1_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved performance metrics: {metrics_file}")

        # 3. Generate and save report
        report_file = output_path / f'backtest_v1_report_{timestamp}.txt'
        report = self.generate_report(metrics)
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved backtest report: {report_file}")

        return trades_file, metrics_file, report_file

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate human-readable performance report

        Args:
            metrics: Performance metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS - STRATEGY V1 (ASX200 SWING TRADING)")
        report.append("=" * 80)
        report.append("")

        # Period
        report.append("BACKTEST PERIOD")
        report.append("-" * 80)
        report.append(f"Start Date:       {metrics.start_date}")
        report.append(f"End Date:         {metrics.end_date}")
        report.append(f"Total Days:       {metrics.total_days}")
        report.append(f"Trading Days:     {metrics.trading_days}")
        report.append("")

        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Trades:     {metrics.total_trades}")
        report.append(f"Winning Trades:   {metrics.winning_trades} ({metrics.win_rate:.1f}%)")
        report.append(f"Losing Trades:    {metrics.losing_trades} ({100-metrics.win_rate:.1f}%)")
        report.append(f"Signals/Week:     {metrics.signals_per_week:.1f}")
        report.append(f"Avg Candidates:   {metrics.avg_candidates_per_scan:.1f} per scan")
        report.append("")

        # Returns
        report.append("RETURNS")
        report.append("-" * 80)
        report.append(f"Average Gain:     +{metrics.avg_gain_pct:.2f}%")
        report.append(f"Average Loss:     {metrics.avg_loss_pct:.2f}%")
        report.append(f"Average Trade:    {metrics.avg_trade_pct:+.2f}%")
        report.append(f"Total Return:     {metrics.total_return_pct:+.2f}%")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 80)
        report.append(f"Profit Factor:    {metrics.profit_factor:.2f}")
        report.append(f"Max Drawdown:     {metrics.max_drawdown_pct:.2f}%")
        report.append(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        report.append(f"Avg Hold Days:    {metrics.avg_hold_days:.1f}")
        report.append(f"Median Hold Days: {metrics.median_hold_days:.0f}")
        report.append("")

        # Entry Type Performance
        report.append("ENTRY TYPE PERFORMANCE")
        report.append("-" * 80)
        report.append(f"{'Entry Type':<20} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)
        for entry_type, perf in sorted(metrics.entry_type_performance.items()):
            report.append(f"{entry_type.replace('_', ' ').title():<20} "
                         f"{perf['count']:>8} "
                         f"{perf['win_rate']:>11.1f}% "
                         f"{perf['avg_return']:>14.2f}%")
        report.append("")

        # Sector Performance
        report.append("SECTOR PERFORMANCE")
        report.append("-" * 80)
        report.append(f"{'Sector':<30} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)
        for sector, perf in sorted(metrics.sector_performance.items(),
                                   key=lambda x: x[1]['count'], reverse=True):
            report.append(f"{sector:<30} "
                         f"{perf['count']:>8} "
                         f"{perf['win_rate']:>11.1f}% "
                         f"{perf['avg_return']:>14.2f}%")
        report.append("")

        # Return Distribution
        report.append("RETURN DISTRIBUTION")
        report.append("-" * 80)
        for bin_range, count in metrics.return_distribution.items():
            pct = (count / metrics.total_trades * 100) if metrics.total_trades > 0 else 0
            bar = '█' * int(pct / 2)
            report.append(f"{bin_range:<15} {count:>4} ({pct:>5.1f}%)  {bar}")
        report.append("")

        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        return "\n".join(report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Backtest v1 Swing Trading Strategy for ASX200',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for last 6 months
  python backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01

  # With verbose logging
  python backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01 --verbose

  # Custom output directory
  python backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01 --output-dir ./results
        """
    )

    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/AU-testing/data/asx200_historical.db',
        help='Path to SQLite database (default: AU-testing/data/asx200_historical.db)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/AU-testing/backtesting/results',
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate dates
    try:
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)

        if start_dt >= end_dt:
            logger.error("Start date must be before end date")
            return

        if (end_dt - start_dt).days < 30:
            logger.warning("Backtest period is less than 30 days - results may not be meaningful")

    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        return

    # Initialize backtester
    backtester = BacktestV1(
        db_path=args.db_path,
        verbose=args.verbose
    )

    # Run backtest
    try:
        trades = backtester.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )

        if not trades:
            logger.warning("No trades generated during backtest period")
            return

        # Calculate metrics
        metrics = backtester.calculate_performance_metrics(
            trades=trades,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Save results
        trades_file, metrics_file, report_file = backtester.save_results(
            trades=trades,
            metrics=metrics,
            output_dir=args.output_dir
        )

        # Print summary to console
        print("\n")
        print(backtester.generate_report(metrics))
        print("\n")
        print(f"Results saved to: {args.output_dir}")
        print(f"  - Trades:  {trades_file.name}")
        print(f"  - Metrics: {metrics_file.name}")
        print(f"  - Report:  {report_file.name}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
