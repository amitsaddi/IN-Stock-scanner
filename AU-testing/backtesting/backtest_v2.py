"""
Comprehensive Backtesting Engine for v2 Enhanced Swing Trading Strategy (ASX200)

This module implements a complete backtesting framework for the PROPOSED v2
swing trading strategy with enhanced filters, scoring, and entry signals.

Strategy v2 Criteria (ENHANCED):
- Fundamental: Market Cap >= A$500M, D/E <= 1.2 (relaxed), ROE >= 8% (relaxed)
- NEW: Daily Volume >= 500k shares (liquidity filter)
- Technical: RSI 30-70 (wider), 52W High 80-100% (no cap), Volume >= 1.0x, ATR >= 1.5%
- NEW: ATR% filter for volatility
- Scoring: Rebalanced (Momentum 30%, Technical 30%, Volume 20%, Fundamental 20%)
- Entry Signals: 5 from v1 + 3 NEW (Bollinger Bounce, Volume Surge, Consolidation Breakout)
- Risk Management: 3-15 day hold, 12-15% targets, 5-7% stop loss, Max 20 candidates

Author: Backtesting Framework
Date: 2025-11-11
Version: 2.0
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
from collections import defaultdict

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

    # v2 specific metrics
    atr_pct_entry: Optional[float] = None
    daily_volume_entry: Optional[float] = None

    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_date is None

    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.profit_loss_pct is not None and self.profit_loss_pct > 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results"""
    # Version identifier
    strategy_version: str

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

    # v2 specific metrics
    atr_filter_impact: Dict[str, float]  # How ATR filter affects results
    liquidity_filter_impact: Dict[str, float]  # How volume filter affects results
    new_signal_performance: Dict[str, Dict]  # Performance of 3 new signals


# ============================================================================
# SECTOR WEIGHTS (V2 - UPDATED FOR 2025)
# ============================================================================

V2_SECTOR_WEIGHTS = {
    'Health Care': 1.2,          # Defensive, outperforming
    'Consumer Staples': 1.15,    # Defensive, stable earnings
    'Materials': 1.1,            # Still strong but moderated
    'Utilities': 1.0,            # Defensive positioning
    'Financials': 1.0,           # Banking stable
    'Consumer Discretionary': 0.95,  # Under pressure
    'Industrials': 0.95,         # Moderate
    'Real Estate': 0.9,          # Slight improvement
    'Energy': 0.9,               # Underperforming
    'Information Technology': 0.85,  # Still weak but potential
    'Communication Services': 0.85   # Neutral
}


# ============================================================================
# ENTRY SIGNAL DEFINITIONS (V2 - 8 SIGNALS)
# ============================================================================

ENTRY_SIGNAL_PARAMS = {
    # Original v1 signals (5)
    'pullback': {'target_pct': 12, 'stop_loss_pct': 7},
    'breakout': {'target_pct': 15, 'stop_loss_pct': 5},
    'macd_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'ma_cross': {'target_pct': 15, 'stop_loss_pct': 6},
    'trend_follow': {'target_pct': 15, 'stop_loss_pct': 6},

    # NEW v2 signals (3)
    'bollinger_bounce': {'target_pct': 12, 'stop_loss_pct': 7},     # Conservative
    'volume_surge': {'target_pct': 15, 'stop_loss_pct': 5},          # Momentum
    'consolidation_breakout': {'target_pct': 15, 'stop_loss_pct': 5} # Breakout
}


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestV2:
    """
    Comprehensive backtesting engine for v2 enhanced swing trading strategy

    Key v2 Enhancements:
    - Relaxed fundamental filters (D/E <= 1.2, ROE >= 8%)
    - NEW liquidity filter (Volume >= 500k)
    - Wider technical ranges (RSI 30-70, 52W 80-100%)
    - NEW ATR filter (>= 1.5%)
    - Rebalanced scoring (Momentum 30%, Technical 30%, Volume 20%, Fundamental 20%)
    - 3 NEW entry signals (Bollinger, Volume Surge, Consolidation)
    - Updated sector weights for 2025
    - Lower minimum score (55 vs 60)
    - More candidates (20 vs 15)
    """

    def __init__(self, db_path: str, verbose: bool = False):
        """
        Initialize v2 backtesting engine

        Args:
            db_path: Path to SQLite database with historical data
            verbose: Enable detailed logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self.trades: List[Trade] = []
        self.trade_counter = 0

        # v2 tracking metrics
        self.atr_filtered_count = 0
        self.liquidity_filtered_count = 0

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
            volume_ratio, week_52_high, week_52_high_proximity,
            atr, bb_upper, bb_middle, bb_lower
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
        - Market Cap >= A$500M (50,000 lakhs) [SAME as v1]
        - Debt-to-Equity <= 1.2 [RELAXED from 1.0]
        - ROE >= 8% [RELAXED from 10%]

        Args:
            metadata: Stock metadata DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered = metadata[
            (metadata['market_cap'] >= 50000) &  # A$500M = 50,000 lakhs
            (metadata['debt_to_equity'] <= 1.2) &  # v2: Relaxed from 1.0
            (metadata['roe'] >= 8.0)  # v2: Relaxed from 10.0
        ].copy()

        logger.debug(f"Fundamental filters (v2): {len(metadata)} -> {len(filtered)} stocks")
        return filtered

    def calculate_atr_percentage(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR as percentage of price (v2 NEW filter)

        Args:
            df: Historical data

        Returns:
            ATR as percentage of current price
        """
        if df.empty:
            return 0

        latest = df.iloc[-1]
        atr = latest.get('atr', 0)
        close = latest['close']

        if close > 0:
            return (atr / close) * 100
        return 0

    def apply_technical_filters(self, df: pd.DataFrame) -> bool:
        """
        Apply v2 technical filters (WIDENED from v1)

        Criteria:
        - RSI: 30-70 [WIDER from 35-65]
        - 52-week high proximity: 80-100% [WIDER from 85-98%, no upper cap]
        - Volume ratio >= 1.0x [LOWER from 1.2x]
        - NEW: ATR >= 1.5% (volatility requirement)
        - NEW: Daily volume >= 500k shares (liquidity requirement)

        Args:
            df: Historical data with latest point to check

        Returns:
            True if passes all technical filters
        """
        if df.empty or len(df) < 1:
            return False

        latest = df.iloc[-1]

        # RSI check (WIDER range)
        rsi = latest.get('rsi', 0)
        if not (30 <= rsi <= 70):  # v2: Wider from 35-65
            return False

        # 52-week high proximity check (WIDER, no upper cap)
        week_52_prox = latest.get('week_52_high_proximity', 0)
        if not (80 <= week_52_prox <= 100):  # v2: Wider from 85-98%
            return False

        # Volume ratio check (LOWER threshold)
        volume_ratio = latest.get('volume_ratio', 0)
        if volume_ratio < 1.0:  # v2: Lower from 1.2
            return False

        # NEW v2: ATR percentage check (volatility filter)
        atr_pct = self.calculate_atr_percentage(df)
        if atr_pct < 1.5:
            self.atr_filtered_count += 1
            return False

        # NEW v2: Daily volume check (liquidity filter)
        volume = latest.get('volume', 0)
        if volume < 500000:  # 500k shares minimum
            self.liquidity_filtered_count += 1
            return False

        return True

    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 technical score (30% of total, DOWN from 35%)

        Components:
        - RSI positioning: 10% (down from ~12%)
        - MACD signals: 10% (down from ~12%)
        - MA alignment: 10% (down from ~11%)

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
        rsi_score = 10 * (1 - abs(rsi - 50) / 50)
        score += max(0, rsi_score)

        # MACD signals (0-10 points)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_hist', 0)

        if macd > macd_signal and macd_hist > 0:
            score += 10
        elif macd > macd_signal:
            score += 5

        # MA alignment (0-10 points)
        close = latest['close']
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)

        if close > ema_20 > ema_50 > sma_200:
            score += 10
        elif close > ema_20 > ema_50:
            score += 7
        elif close > ema_20:
            score += 4

        return min(30, score)

    def calculate_fundamental_score(self, market_cap: float, roe: float, debt_equity: float) -> float:
        """
        Calculate v2 fundamental score (20% of total, DOWN from 25%)

        Components:
        - ROE: 8% (down from 10%)
        - Debt/Equity: 7% (down from 8%)
        - Market Cap: 5% (down from 7%)

        Args:
            market_cap: Market cap in lakhs
            roe: Return on equity %
            debt_equity: Debt-to-equity ratio

        Returns:
            Fundamental score (0-20)
        """
        score = 0

        # ROE score (0-8 points) - normalize around 8-28%
        roe_normalized = min((roe - 8) / 20, 1.0)
        score += 8 * max(0, roe_normalized)

        # Debt/Equity score (0-7 points) - lower is better, cap at 1.2
        de_score = (1.2 - min(debt_equity, 1.2)) / 1.2 * 7
        score += max(0, de_score)

        # Market cap score (0-5 points) - logarithmic scale
        if market_cap >= 50000:
            mcap_normalized = min(np.log10(market_cap / 50000) / np.log10(10), 1.0)
            score += 5 * mcap_normalized

        return min(20, score)

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 momentum score (30% of total, UP from 25%)

        Components:
        - 52-week high proximity: 15% (increased)
        - Price action strength: 10% (new)
        - Trend consistency: 5% (new)

        Args:
            df: Historical data

        Returns:
            Momentum score (0-30)
        """
        if df.empty or len(df) < 10:
            return 0

        latest = df.iloc[-1]
        score = 0

        # 52-week high proximity (0-15 points)
        week_52_prox = latest.get('week_52_high_proximity', 0)
        # Linear scoring: 80% = 0 points, 100% = 15 points
        prox_score = ((week_52_prox - 80) / 20) * 15
        score += max(0, min(15, prox_score))

        # Price action strength (0-10 points) - 5-day price change
        if len(df) >= 5:
            price_5d_ago = df.iloc[-5]['close']
            current_price = latest['close']
            price_change_pct = ((current_price / price_5d_ago) - 1) * 100

            # Reward positive momentum, cap at +10%
            momentum_score = min(price_change_pct / 10, 1.0) * 10
            score += max(0, momentum_score)

        # Trend consistency (0-5 points) - count of closes above EMA(20)
        if len(df) >= 10:
            last_10 = df.tail(10)
            closes_above_ema = sum(1 for _, row in last_10.iterrows()
                                  if row['close'] > row.get('ema_20', 0))
            consistency_score = (closes_above_ema / 10) * 5
            score += consistency_score

        return min(30, score)

    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calculate v2 volume score (20% of total, UP from 15%)

        Components:
        - Volume ratio: 10% (increased)
        - Volume trend: 5% (new)
        - Liquidity: 5% (new)

        Args:
            df: Historical data

        Returns:
            Volume score (0-20)
        """
        if df.empty or len(df) < 5:
            return 0

        latest = df.iloc[-1]
        score = 0

        # Volume ratio (0-10 points)
        volume_ratio = latest.get('volume_ratio', 0)
        # 1.0x = 0 points, 2.0x+ = 10 points
        ratio_score = ((volume_ratio - 1.0) / 1.0) * 10
        score += max(0, min(10, ratio_score))

        # Volume trend (0-5 points) - increasing volume over 5 days
        if len(df) >= 5:
            last_5_volumes = df.tail(5)['volume'].values
            # Check if generally trending up (simple regression)
            x = np.arange(5)
            slope = np.polyfit(x, last_5_volumes, 1)[0]
            if slope > 0:
                # Normalize slope relative to average volume
                avg_volume = np.mean(last_5_volumes)
                trend_strength = min((slope * 5) / avg_volume, 1.0)  # 5-day change
                score += trend_strength * 5

        # Liquidity score (0-5 points) - based on absolute volume
        volume = latest.get('volume', 0)
        # 500k = 0 points, 2M+ = 5 points
        liquidity_score = min((volume - 500000) / 1500000, 1.0) * 5
        score += max(0, liquidity_score)

        return min(20, score)

    def calculate_total_score(self, df: pd.DataFrame, market_cap: float,
                             roe: float, debt_equity: float, sector: str) -> float:
        """
        Calculate total v2 score with sector weighting

        v2 Scoring Rebalanced:
        - Momentum: 30% (up from 25%)
        - Technical: 30% (down from 35%)
        - Volume: 20% (up from 15%)
        - Fundamental: 20% (down from 25%)

        Args:
            df: Historical data
            market_cap: Market cap in lakhs
            roe: Return on equity
            debt_equity: Debt-to-equity ratio
            sector: Stock sector

        Returns:
            Total score (0-100+, sector weighted)
        """
        momentum_score = self.calculate_momentum_score(df)
        tech_score = self.calculate_technical_score(df)
        volume_score = self.calculate_volume_score(df)
        fund_score = self.calculate_fundamental_score(market_cap, roe, debt_equity)

        base_score = momentum_score + tech_score + volume_score + fund_score

        # Apply v2 sector weight (updated for 2025)
        sector_weight = V2_SECTOR_WEIGHTS.get(sector, 1.0)
        final_score = base_score * sector_weight

        return final_score

    def check_consolidation_pattern(self, df: pd.DataFrame) -> bool:
        """
        Check for consolidation pattern (v2 NEW signal)

        Pattern: Trading in tight range (< 5% over 5+ days)

        Args:
            df: Historical data (need at least 5 days)

        Returns:
            True if consolidation pattern detected
        """
        if len(df) < 6:  # Need 5 days + current
            return False

        # Look at previous 5 days (not including current)
        lookback = df.iloc[-6:-1]

        high_5d = lookback['high'].max()
        low_5d = lookback['low'].min()

        # Calculate range as percentage
        range_pct = ((high_5d - low_5d) / low_5d) * 100

        return range_pct < 5.0

    def identify_entry_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify entry signal type based on v2 criteria (8 signals total)

        v1 Entry Signals (5):
        1. Pullback: RSI 30-35, near support
        2. Breakout: Price > EMA_20 with volume
        3. MACD Cross: Fresh bullish crossover
        4. MA Cross: EMA(20) crosses above EMA(50)
        5. Trend Follow: Above all MAs, near 52W high

        v2 NEW Entry Signals (3):
        6. Bollinger Bounce: Price touches lower BB + RSI < 35
        7. Volume Surge: Volume > 2x + Price up > 3%
        8. Consolidation Breakout: Tight range < 5% for 5+ days, then breakout

        Args:
            df: Historical data (need at least 6 days for new signals)

        Returns:
            Entry signal type or None
        """
        if df.empty or len(df) < 2:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest.get('rsi', 50)
        close = latest['close']
        prev_close = prev['close']
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)
        sma_200 = latest.get('sma_200', 0)
        volume_ratio = latest.get('volume_ratio', 1.0)
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = prev.get('macd_hist', 0)
        week_52_prox = latest.get('week_52_high_proximity', 0)
        bb_lower = latest.get('bb_lower', 0)
        bb_upper = latest.get('bb_upper', 0)

        # Priority order for signal detection

        # 1. NEW v2: Bollinger Bounce (oversold at lower band)
        if bb_lower > 0:
            price_near_lower = abs(close - bb_lower) / bb_lower < 0.02  # Within 2%
            if price_near_lower and rsi < 35:
                return 'bollinger_bounce'

        # 2. NEW v2: Volume Surge (institutional interest)
        price_change_pct = ((close / prev_close) - 1) * 100 if prev_close > 0 else 0
        if volume_ratio > 2.0 and price_change_pct > 3.0:
            return 'volume_surge'

        # 3. Pullback (highest priority for mean reversion from v1)
        if 30 <= rsi <= 35 and close < ema_20:
            return 'pullback'

        # 4. MACD Cross (fresh bullish crossover)
        if macd_hist > 0 and prev_macd_hist <= 0:
            return 'macd_cross'

        # 5. NEW v2: Consolidation Breakout
        if len(df) >= 6:
            is_consolidation = self.check_consolidation_pattern(df)
            breakout_confirmed = close > prev_close * 1.02 and volume_ratio > 1.3
            if is_consolidation and breakout_confirmed:
                return 'consolidation_breakout'

        # 6. MA Cross (EMA 20 just crossed EMA 50)
        prev_ema_20 = prev.get('ema_20', 0)
        prev_ema_50 = prev.get('ema_50', 0)
        if ema_20 > ema_50 and prev_ema_20 <= prev_ema_50:
            return 'ma_cross'

        # 7. Breakout (price breaks above EMA 20 with volume)
        if close > ema_20 * 1.02 and volume_ratio > 1.5:
            return 'breakout'

        # 8. Trend Follow (above all MAs, near 52W high)
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
        Simulate a single day's scan with v2 criteria

        Args:
            scan_date: Date to scan (YYYY-MM-DD)
            metadata: Stock metadata with fundamentals

        Returns:
            List of candidate dictionaries with scores
        """
        candidates = []

        # Apply v2 fundamental filters (relaxed)
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

            # Apply v2 technical filters (wider ranges + new filters)
            if not self.apply_technical_filters(df):
                continue

            # Calculate v2 score (rebalanced)
            score = self.calculate_total_score(df, market_cap, roe, debt_equity, sector)

            # Check v2 minimum score threshold (55, down from 60)
            if score < 55:
                continue

            # Identify entry signal (8 types including 3 new)
            entry_signal = self.identify_entry_signal(df)
            if entry_signal is None:
                continue

            latest = df.iloc[-1]
            atr_pct = self.calculate_atr_percentage(df)

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

        # Sort by score and take top 20 (v2: increased from 15)
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
        Run complete v2 backtest simulation

        Simulates daily scans with v2 criteria, tracks trades, and calculates performance

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            List of all trades executed
        """
        logger.info("=" * 80)
        logger.info(f"Starting v2 ENHANCED Strategy Backtest: {start_date} to {end_date}")
        logger.info("=" * 80)

        self.connect_db()

        # Reset v2 tracking metrics
        self.atr_filtered_count = 0
        self.liquidity_filtered_count = 0

        # Get stock metadata
        metadata = self.get_stock_metadata()

        # Generate list of trading dates
        date_range = pd.bdate_range(start=start_date, end=end_date)
        total_dates = len(date_range)

        logger.info(f"Total trading days to simulate: {total_dates}")
        logger.info(f"Stock universe: {len(metadata)} stocks")
        logger.info(f"v2 Enhancements: Wider ranges, New filters (ATR, Volume), 3 new signals")

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
        logger.info(f"v2 Backtest Complete")
        logger.info(f"Total trades: {len(closed_trades)}")
        logger.info(f"Average candidates per scan: {np.mean(scan_results_count):.1f}")
        logger.info(f"ATR filter rejections: {self.atr_filtered_count}")
        logger.info(f"Liquidity filter rejections: {self.liquidity_filtered_count}")
        logger.info("=" * 80)

        self.close_db()
        return closed_trades

    def calculate_performance_metrics(self, trades: List[Trade],
                                     start_date: str, end_date: str) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for v2 strategy

        Args:
            trades: List of completed trades
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            PerformanceMetrics object with v2-specific metrics
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

        # Entry type breakdown (all 8 signals including 3 new)
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

        # v2 SPECIFIC METRICS

        # ATR filter impact
        atr_values = [t.atr_pct_entry for t in trades if t.atr_pct_entry]
        atr_filter_impact = {
            'avg_atr_pct': np.mean(atr_values) if atr_values else 0,
            'min_atr_pct': np.min(atr_values) if atr_values else 0,
            'max_atr_pct': np.max(atr_values) if atr_values else 0,
            'filtered_out': self.atr_filtered_count
        }

        # Liquidity filter impact
        volume_values = [t.daily_volume_entry for t in trades if t.daily_volume_entry]
        liquidity_filter_impact = {
            'avg_volume': np.mean(volume_values) if volume_values else 0,
            'min_volume': np.min(volume_values) if volume_values else 0,
            'max_volume': np.max(volume_values) if volume_values else 0,
            'filtered_out': self.liquidity_filtered_count
        }

        # Performance of 3 NEW signals
        new_signals = ['bollinger_bounce', 'volume_surge', 'consolidation_breakout']
        new_signal_performance = {}
        for signal in new_signals:
            signal_trades = [t for t in trades if t.entry_signal == signal]
            if signal_trades:
                signal_wins = sum(1 for t in signal_trades if t.is_winner())
                new_signal_performance[signal] = {
                    'count': len(signal_trades),
                    'win_rate': (signal_wins / len(signal_trades) * 100),
                    'avg_return': np.mean([t.profit_loss_pct for t in signal_trades])
                }
            else:
                new_signal_performance[signal] = {
                    'count': 0,
                    'win_rate': 0,
                    'avg_return': 0
                }

        metrics = PerformanceMetrics(
            strategy_version='v2',
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
            return_distribution=return_distribution,
            atr_filter_impact=atr_filter_impact,
            liquidity_filter_impact=liquidity_filter_impact,
            new_signal_performance=new_signal_performance
        )

        return metrics

    def save_results(self, trades: List[Trade], metrics: PerformanceMetrics,
                    output_dir: str):
        """
        Save v2 backtest results to files

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
        trades_file = output_path / f'backtest_v2_trades_{timestamp}.csv'
        trades_df = pd.DataFrame([asdict(t) for t in trades])
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved detailed trades: {trades_file}")

        # 2. Save performance metrics JSON
        metrics_file = output_path / f'backtest_v2_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved performance metrics: {metrics_file}")

        # 3. Generate and save report
        report_file = output_path / f'backtest_v2_report_{timestamp}.txt'
        report = self.generate_report(metrics)
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved backtest report: {report_file}")

        return trades_file, metrics_file, report_file

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate human-readable performance report for v2 strategy

        Args:
            metrics: Performance metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS - STRATEGY V2 (ENHANCED ASX200 SWING TRADING)")
        report.append("=" * 80)
        report.append("")

        # v2 Enhancements Summary
        report.append("V2 STRATEGY ENHANCEMENTS")
        report.append("-" * 80)
        report.append("Fundamental: D/E <= 1.2 (relaxed), ROE >= 8% (relaxed)")
        report.append("Technical:   RSI 30-70 (wider), 52W 80-100% (no cap), Volume >= 1.0x")
        report.append("NEW Filters: ATR >= 1.5%, Daily Volume >= 500k shares")
        report.append("Scoring:     Momentum 30%, Technical 30%, Volume 20%, Fundamental 20%")
        report.append("Signals:     8 total (5 from v1 + 3 NEW)")
        report.append("Candidates:  Max 20 (increased from 15)")
        report.append("Min Score:   55 (lowered from 60)")
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

        # v2 SPECIFIC: Filter Impact
        report.append("V2 FILTER IMPACT")
        report.append("-" * 80)
        report.append(f"ATR Filter:")
        report.append(f"  Filtered Out:   {metrics.atr_filter_impact['filtered_out']} candidates")
        report.append(f"  Avg ATR%:       {metrics.atr_filter_impact['avg_atr_pct']:.2f}%")
        report.append(f"  Range:          {metrics.atr_filter_impact['min_atr_pct']:.2f}% - "
                     f"{metrics.atr_filter_impact['max_atr_pct']:.2f}%")
        report.append("")
        report.append(f"Liquidity Filter:")
        report.append(f"  Filtered Out:   {metrics.liquidity_filter_impact['filtered_out']} candidates")
        report.append(f"  Avg Volume:     {metrics.liquidity_filter_impact['avg_volume']:,.0f} shares")
        report.append(f"  Range:          {metrics.liquidity_filter_impact['min_volume']:,.0f} - "
                     f"{metrics.liquidity_filter_impact['max_volume']:,.0f}")
        report.append("")

        # v2 SPECIFIC: New Signal Performance
        report.append("V2 NEW ENTRY SIGNALS PERFORMANCE")
        report.append("-" * 80)
        report.append(f"{'Signal':<25} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)
        for signal, perf in metrics.new_signal_performance.items():
            signal_name = signal.replace('_', ' ').title()
            if perf['count'] > 0:
                report.append(f"{signal_name:<25} "
                             f"{perf['count']:>8} "
                             f"{perf['win_rate']:>11.1f}% "
                             f"{perf['avg_return']:>14.2f}%")
            else:
                report.append(f"{signal_name:<25} {'No trades':>8}")
        report.append("")

        # Entry Type Performance (All 8)
        report.append("ALL ENTRY TYPE PERFORMANCE")
        report.append("-" * 80)
        report.append(f"{'Entry Type':<25} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)
        for entry_type, perf in sorted(metrics.entry_type_performance.items(),
                                       key=lambda x: x[1]['count'], reverse=True):
            report.append(f"{entry_type.replace('_', ' ').title():<25} "
                         f"{perf['count']:>8} "
                         f"{perf['win_rate']:>11.1f}% "
                         f"{perf['avg_return']:>14.2f}%")
        report.append("")

        # Sector Performance
        report.append("SECTOR PERFORMANCE (V2 2025 WEIGHTS)")
        report.append("-" * 80)
        report.append(f"{'Sector':<30} {'Weight':>8} {'Count':>8} {'Win Rate':>12} {'Avg Return':>15}")
        report.append("-" * 80)
        for sector, perf in sorted(metrics.sector_performance.items(),
                                   key=lambda x: x[1]['count'], reverse=True):
            weight = V2_SECTOR_WEIGHTS.get(sector, 1.0)
            report.append(f"{sector:<30} "
                         f"{weight:>7.2f}x "
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
        report.append("Strategy Version: v2 (Enhanced)")
        report.append("=" * 80)

        return "\n".join(report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Backtest v2 ENHANCED Swing Trading Strategy for ASX200',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v2 Strategy Enhancements:
  - Relaxed fundamental filters (D/E <= 1.2, ROE >= 8%)
  - NEW liquidity filter (Volume >= 500k shares)
  - Wider technical ranges (RSI 30-70, 52W 80-100%)
  - NEW ATR filter (>= 1.5%)
  - Rebalanced scoring weights
  - 3 NEW entry signals (Bollinger Bounce, Volume Surge, Consolidation Breakout)
  - Updated sector weights for 2025
  - Lower minimum score (55 vs 60)
  - More candidates (20 vs 15)

Examples:
  # Run v2 backtest for last 6 months
  python backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01

  # With verbose logging
  python backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01 --verbose

  # Custom output directory
  python backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01 --output-dir ./results
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

    # Initialize v2 backtester
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
