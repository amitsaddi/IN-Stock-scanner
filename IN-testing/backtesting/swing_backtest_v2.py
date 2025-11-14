"""
Comprehensive Backtesting Engine for v2 Swing Trading Strategy (Indian Market - Nifty 500)

This module implements a complete backtesting framework for the enhanced v2
swing trading strategy, simulating daily scans over historical data and tracking
trade performance metrics.

Strategy v2 Criteria (Enhanced from v1):
- Fundamental: Market Cap >= ₹5000 Cr, D/E <= 0.8 (relaxed), ROE >= 12% (relaxed)
- Technical: RSI 30-70 (wider), Volume >= 1.0x (relaxed), ATR >= 1.5%, Liquidity >= 500k
- Entry Signals: 5 original + 3 NEW (Bollinger Bounce, Volume Surge, Consolidation Breakout)
- Risk Management: 3-15 day hold, 12-15% targets, 5-7% stop loss
- Max Candidates: 20 (increased from 15)
- Min Score: 55 (lowered from 60)

Changes from v1:
1. Relaxed fundamental filters (D/E: 0.5→0.8, ROE: 15%→12%)
2. Wider RSI range (40-60 → 30-70)
3. Relaxed volume requirement (1.2x → 1.0x)
4. NEW ATR filter (min 1.5% volatility)
5. NEW Liquidity filter (min 500k daily volume)
6. THREE new entry signals
7. Adjusted scoring weights (more momentum/technical focus)

Author: Backtesting Framework
Date: 2025-11-11
Market: India (NSE)
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
    atr_pct_entry: Optional[float] = None  # NEW for v2
    daily_volume_entry: Optional[float] = None  # NEW for v2

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
# SECTOR WEIGHTS (V2 - INDIA NOV 2025)
# ============================================================================

V2_SECTOR_WEIGHTS = {
    'DEFENCE': 1.1,
    'CAPITAL GOODS': 1.1,
    'INFRASTRUCTURE': 1.1,
    'PSU BANKS': 1.1,
    'RENEWABLE ENERGY': 1.1,
    'FINANCIALS': 1.0,
    'AUTO': 1.0,
    'CONSUMER DURABLES': 1.0,
    'POWER': 0.95,
    'METALS': 0.95,
    'IT': 0.9,
    'PHARMA': 0.9,
    'FMCG': 0.95,
    'CEMENT': 1.0,
    'TELECOM': 0.95
}


# ============================================================================
# ENTRY SIGNAL DEFINITIONS (V2 - INCLUDES 3 NEW SIGNALS)
# ============================================================================

ENTRY_SIGNAL_PARAMS = {
    # Original 5 signals
    'pullback': {'target_pct': 12, 'stop_loss_pct': 7},
    'breakout': {'target_pct': 15, 'stop_loss_pct': 5},
    'macd_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'ma_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'trend_follow': {'target_pct': 15, 'stop_loss_pct': 6},
    # NEW v2 signals
    'bollinger_bounce': {'target_pct': 12, 'stop_loss_pct': 6},
    'volume_surge': {'target_pct': 13, 'stop_loss_pct': 6},
    'consolidation_breakout': {'target_pct': 14, 'stop_loss_pct': 5}
}


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestV2:
    """
    Comprehensive backtesting engine for v2 swing trading strategy (India)
    Enhanced version with relaxed filters and new signals
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
            symbol: Stock symbol (without .NS)
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
            volume_ratio, week_52_high, week_52_high_proximity,
            bb_upper, bb_middle, bb_lower, atr
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
        Apply v2 fundamental filters (RELAXED from v1)

        Criteria:
        - Market Cap >= ₹5000 Cr (same)
        - Debt-to-Equity <= 0.8 (RELAXED from 0.5)
        - ROE >= 12% (RELAXED from 15%)

        Args:
            metadata: Stock metadata DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered = metadata[
            (metadata['market_cap'] >= 5000) &  # ₹5000 Cr
            (metadata['debt_to_equity'] <= 0.8) &  # RELAXED from v1's 0.5
            (metadata['roe'] >= 12.0)  # RELAXED from v1's 15.0
        ].copy()

        logger.debug(f"Fundamental filters (v2): {len(metadata)} -> {len(filtered)} stocks")
        return filtered

    def apply_technical_filters(self, df: pd.DataFrame) -> bool:
        """
        Apply v2 technical filters to latest data point (ENHANCED from v1)

        Criteria:
        - RSI: 30-70 (WIDER than v1's 40-60)
        - Volume ratio >= 1.0x (RELAXED from v1's 1.2x)
        - ATR >= 1.5% (NEW for v2)
        - Daily volume >= 500k shares (NEW for v2)

        Args:
            df: Historical data with latest point to check

        Returns:
            True if passes all technical filters
        """
        if df.empty or len(df) < 1:
            return False

        latest = df.iloc[-1]

        # RSI check (WIDER range than v1)
        rsi = latest.get('rsi', 0)
        if not (30 <= rsi <= 70):  # v1 was 40-60
            return False

        # Volume ratio check (RELAXED from v1)
        volume_ratio = latest.get('volume_ratio', 0)
        if volume_ratio < 1.0:  # v1 was 1.2
            return False

        # NEW: ATR filter (minimum volatility)
        atr = latest.get('atr', 0)
        close = latest.get('close', 0)
        if close > 0:
            atr_pct = (atr / close) * 100
            if atr_pct < 1.5:  # Minimum 1.5% volatility
                return False

        # NEW: Liquidity filter (minimum daily volume)
        volume = latest.get('volume', 0)
        if volume < 500000:  # Minimum 500k shares
            return False

        return True

    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 technical score (30 points - INCREASED from v1's 20)

        Components:
        - RSI positioning (optimal at 50)
        - MACD signals
        - MA alignment
        - Bollinger Bands positioning

        Args:
            df: Historical data

        Returns:
            Technical score (0-30)
        """
        if df.empty or len(df) < 2:
            return 0

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0

        # RSI positioning (0-10 points) - optimal at 50
        rsi = latest.get('rsi', 50)
        if 30 <= rsi <= 70:
            # Wider range but still reward middle values
            rsi_score = 10 * (1 - abs(rsi - 50) / 20)
            score += max(0, rsi_score)

        # MACD signals (0-10 points)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_hist', 0)

        if macd > macd_signal and macd_hist > 0:
            score += 10
        elif macd > macd_signal:
            score += 5

        # MA alignment (0-7 points)
        close = latest['close']
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)

        if close > ema_20 > ema_50 > sma_200:
            score += 7
        elif close > ema_20 > ema_50:
            score += 4
        elif close > ema_20:
            score += 2

        # NEW: Bollinger positioning (0-3 points)
        bb_upper = latest.get('bb_upper', 0)
        bb_lower = latest.get('bb_lower', 0)
        if bb_upper > 0 and bb_lower > 0:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            if 0.3 <= bb_position <= 0.7:  # Middle range
                score += 3
            elif bb_position < 0.3:  # Near lower (potential bounce)
                score += 2

        return min(30, score)

    def calculate_fundamental_score(self, market_cap: float, roe: float, debt_equity: float) -> float:
        """
        Calculate v2 fundamental score (20 points - REBALANCED weights)

        Components:
        - ROE (higher is better)
        - Debt/Equity (lower is better)
        - Market Cap (larger is better)

        Args:
            market_cap: Market cap in crores
            roe: Return on equity %
            debt_equity: Debt-to-equity ratio

        Returns:
            Fundamental score (0-20)
        """
        score = 0

        # ROE score (0-8 points) - normalize around 12-35% (relaxed baseline)
        if roe >= 12:
            roe_normalized = min((roe - 12) / 23, 1.0)
            score += 8 * roe_normalized

        # Debt/Equity score (0-7 points) - relaxed cap to 0.8
        de_score = (0.8 - min(debt_equity, 0.8)) / 0.8 * 7
        score += max(0, de_score)

        # Market cap score (0-5 points) - logarithmic scale
        if market_cap >= 5000:
            mcap_normalized = min(np.log10(market_cap / 5000) / np.log10(10), 1.0)
            score += 5 * mcap_normalized

        return min(20, score)

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 momentum score (30 points - INCREASED from v1's 20)

        Components:
        - Price action
        - Trend strength
        - Volume momentum

        Args:
            df: Historical data

        Returns:
            Momentum score (0-30)
        """
        if df.empty or len(df) < 5:
            return 0

        latest = df.iloc[-1]
        score = 0

        # Price momentum (0-15 points) - increased weight
        if len(df) >= 5:
            price_5d_ago = df.iloc[-5]['close']
            current_price = latest['close']
            price_change_pct = ((current_price - price_5d_ago) / price_5d_ago) * 100

            if price_change_pct > 0:
                score += min(15, price_change_pct * 2.5)

        # MA trend strength (0-15 points) - increased weight
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        close = latest['close']

        if ema_20 > 0 and ema_50 > 0:
            dist_20 = ((close - ema_20) / ema_20) * 100
            dist_50 = ((close - ema_50) / ema_50) * 100

            if dist_20 > 0 and dist_50 > 0:
                score += min(15, (dist_20 + dist_50) * 2.5)

        return min(30, score)

    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 volume score (20 points - same as v1)

        Component:
        - Volume ratio vs 20-day average

        Args:
            df: Historical data

        Returns:
            Volume score (0-20)
        """
        if df.empty:
            return 0

        latest = df.iloc[-1]
        volume_ratio = latest.get('volume_ratio', 0)

        # 1.0x = 0 points, 2.5x+ = 20 points (relaxed baseline)
        if volume_ratio >= 1.0:
            score = ((volume_ratio - 1.0) / 1.5) * 20
            return min(20, score)

        return 0

    def calculate_total_score(self, df: pd.DataFrame, market_cap: float,
                             roe: float, debt_equity: float, sector: str) -> float:
        """
        Calculate total v2 score with sector weighting

        Scoring breakdown (v2 - rebalanced):
        - Technical: 30 points (increased from v1's 20)
        - Momentum: 30 points (increased from v1's 20)
        - Volume: 20 points (same)
        - Fundamental: 20 points (decreased from v1's 40)
        Total: 100 points, then apply sector weight

        Args:
            df: Historical data
            market_cap: Market cap in crores
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
        sector_weight = V2_SECTOR_WEIGHTS.get(sector.upper(), 1.0)
        final_score = base_score * sector_weight

        return final_score

    def identify_entry_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify entry signal type based on v2 criteria

        Entry Signals (8 total):
        Original 5:
        1. PULLBACK: RSI 30-40, price near/below 20 EMA
        2. BREAKOUT: Price > 52W high proximity 90-98%, volume >1.5x
        3. MACD_CROSS: Fresh bullish crossover
        4. MA_CROSS: 20 EMA crosses above 50 EMA
        5. TREND_FOLLOW: Above all MAs, strong momentum

        NEW 3 (v2):
        6. BOLLINGER_BOUNCE: Price near/below lower BB, RSI < 40, high volume
        7. VOLUME_SURGE: 2x volume, green candle, above EMA_20
        8. CONSOLIDATION_BREAKOUT: Low ATR period (5d) then breakout with volume

        Args:
            df: Historical data (need at least 6 days for consolidation check)

        Returns:
            Entry signal type or None
        """
        if df.empty or len(df) < 2:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest.get('rsi', 50)
        close = latest['close']
        open_price = latest.get('open', close)
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)
        volume = latest.get('volume', 0)
        volume_ratio = latest.get('volume_ratio', 1.0)
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = prev.get('macd_hist', 0)
        week_52_prox = latest.get('week_52_high_proximity', 0)
        bb_lower = latest.get('bb_lower', 0)
        bb_upper = latest.get('bb_upper', 0)
        atr = latest.get('atr', 0)

        # Calculate average volume (last 20 days)
        if len(df) >= 20:
            avg_volume = df.iloc[-20:]['volume'].mean()
        else:
            avg_volume = volume

        # Priority order for signal detection (NEW signals checked first)

        # NEW 6: Bollinger Bounce
        if bb_lower > 0:
            if (close <= bb_lower * 1.02 and  # Near or below lower band
                rsi < 40 and  # Oversold
                volume > avg_volume * 1.5):  # High volume
                return 'bollinger_bounce'

        # NEW 7: Volume Surge
        if (volume > avg_volume * 2.0 and  # 2x average volume
            close > open_price and  # Green candle
            close > ema_20):  # Above trend
            return 'volume_surge'

        # NEW 8: Consolidation Breakout
        if len(df) >= 6:
            recent_5d = df.iloc[-6:-1]  # Last 5 days before today
            recent_atr_pct = (recent_5d['atr'] / recent_5d['close'] * 100).mean()
            prev_high = recent_5d['high'].max()

            if (recent_atr_pct < 2.0 and  # Low volatility period
                close > prev_high and  # Breakout above recent range
                volume > avg_volume * 1.3):  # Volume confirmation
                return 'consolidation_breakout'

        # Original signals (same priority as v1)

        # 1. Pullback (wider RSI range)
        if 30 <= rsi <= 40 and close <= ema_20 * 1.02:  # RSI range widened
            return 'pullback'

        # 2. MACD Cross
        if macd_hist > 0 and prev_macd_hist <= 0:
            return 'macd_cross'

        # 3. MA Cross
        prev_ema_20 = prev.get('ema_20', 0)
        prev_ema_50 = prev.get('ema_50', 0)
        if ema_20 > ema_50 and prev_ema_20 <= prev_ema_50:
            return 'ma_cross'

        # 4. Breakout
        if week_52_prox >= 90 and volume_ratio > 1.5:
            return 'breakout'

        # 5. Trend Follow
        if close > ema_20 > ema_50 > sma_200 and rsi >= 50:
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

            # Get historical data up to scan date
            lookback_date = (pd.to_datetime(scan_date) - timedelta(days=100)).strftime('%Y-%m-%d')
            df = self.get_historical_data(symbol, lookback_date, scan_date)

            if df.empty or len(df) < 50:
                continue

            # Apply technical filters
            if not self.apply_technical_filters(df):
                continue

            # Calculate score
            score = self.calculate_total_score(df, market_cap, roe, debt_equity, sector)

            # Check minimum score threshold (55 for v2 - LOWERED from v1's 60)
            if score < 55:
                continue

            # Identify entry signal
            entry_signal = self.identify_entry_signal(df)
            if entry_signal is None:
                continue

            latest = df.iloc[-1]

            # Calculate ATR percentage and daily volume for tracking
            atr_pct = (latest.get('atr', 0) / latest['close'] * 100) if latest['close'] > 0 else 0

            candidates.append({
                'symbol': symbol,
                'sector': sector,
                'score': score,
                'entry_signal': entry_signal,
                'entry_price': latest['close'],
                'rsi': latest.get('rsi', 0),
                'volume_ratio': latest.get('volume_ratio', 0),
                'week_52_high_proximity': latest.get('week_52_high_proximity', 0),
                'atr_pct': atr_pct,
                'daily_volume': latest.get('volume', 0)
            })

        # Sort by score and take top 20 (v2 max - INCREASED from v1's 15)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:20]

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
        logger.info(f"Starting v2 Strategy Backtest (India): {start_date} to {end_date}")
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
                    week_52_high_proximity=candidate['week_52_high_proximity'],
                    atr_pct_entry=candidate['atr_pct'],
                    daily_volume_entry=candidate['daily_volume']
                )

                open_trades.append(trade)
                open_symbols.add(candidate['symbol'])

                if self.verbose:
                    logger.debug(f"OPEN: {trade.symbol} | {signal_type.upper()} | "
                               f"Entry: ₹{entry_price:.2f} | Target: ₹{target:.2f} | SL: ₹{stop_loss:.2f}")

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

        # Sharpe ratio
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
        trades_file = output_path / f'swing_backtest_v2_trades_{timestamp}.csv'
        trades_df = pd.DataFrame([asdict(t) for t in trades])
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved detailed trades: {trades_file}")

        # 2. Save performance metrics JSON
        metrics_file = output_path / f'swing_backtest_v2_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved performance metrics: {metrics_file}")

        # 3. Generate and save report
        report_file = output_path / f'swing_backtest_v2_report_{timestamp}.txt'
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
        report.append("BACKTEST RESULTS - STRATEGY V2 (INDIA SWING TRADING - NIFTY 500)")
        report.append("ENHANCED VERSION WITH RELAXED FILTERS AND NEW SIGNALS")
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

        # Entry Type Performance (v2 includes 3 new signals)
        report.append("ENTRY TYPE PERFORMANCE (8 SIGNALS - 3 NEW IN V2)")
        report.append("-" * 80)
        report.append(f"{'Entry Type':<25} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)

        # Mark new signals
        new_signals = {'bollinger_bounce', 'volume_surge', 'consolidation_breakout'}

        for entry_type, perf in sorted(metrics.entry_type_performance.items()):
            marker = " [NEW]" if entry_type in new_signals else ""
            report.append(f"{entry_type.replace('_', ' ').title():<25}{marker:>0} "
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
        report.append("V2 ENHANCEMENTS:")
        report.append("- Relaxed fundamentals: D/E 0.5->0.8, ROE 15%->12%")
        report.append("- Wider RSI range: 40-60 -> 30-70")
        report.append("- Relaxed volume: 1.2x -> 1.0x")
        report.append("- NEW ATR filter: min 1.5% volatility")
        report.append("- NEW Liquidity filter: min 500k shares")
        report.append("- 3 NEW entry signals: Bollinger Bounce, Volume Surge, Consolidation Breakout")
        report.append("- Increased max candidates: 15 -> 20")
        report.append("- Lowered min score: 60 -> 55")
        report.append("- Rebalanced scoring: more momentum/technical focus")
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
        description='Backtest v2 Swing Trading Strategy for Indian Market (Nifty 500) - Enhanced',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for last 6 months
  python swing_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01

  # With verbose logging
  python swing_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01 --verbose

  # Custom output directory
  python swing_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01 --output-dir ./results
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
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/data/nifty500_historical.db',
        help='Path to SQLite database (default: IN-testing/data/nifty500_historical.db)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/backtesting/results',
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
    backtester = BacktestV2(
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
