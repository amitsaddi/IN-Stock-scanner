"""
BTST (Buy Today Sell Tomorrow) Backtesting Engine v1 (Indian Market - Nifty 500)

This module implements a complete backtesting framework for BTST trading strategy,
simulating end-of-day scans (3:15 PM) and next-morning exits (9:15 AM).

BTST v1 Strategy Criteria:
- Entry: 3:15 PM (market close)
- Exit: 9:15 AM next trading day (market open)
- Hold Period: Always 1 day
- Entry Criteria:
  * Day gain: 2.0-3.5%
  * Volume: >= 1.5x average
  * High proximity: >= 90%
  * Above 20 EMA
  * Positive MACD & RSI > 50
- Min Score: 60
- Max Candidates: 10 per day

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
    """Represents a single BTST trade with entry/exit details"""
    trade_id: int
    symbol: str
    sector: str
    entry_date: str
    entry_price: float  # Close price at 3:15 PM
    score: float

    # Exit details (next morning)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None  # Open price at 9:15 AM

    # BTST metrics
    hold_days: int = 1  # Always 1 for BTST
    profit_loss_pct: Optional[float] = None
    profit_loss_abs: Optional[float] = None
    gap_pct: Optional[float] = None  # Gap up/down percentage

    # Entry indicators
    day_gain_pct: Optional[float] = None
    volume_ratio_entry: Optional[float] = None
    high_proximity_pct: Optional[float] = None
    rsi_entry: Optional[float] = None

    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.profit_loss_pct is not None and self.profit_loss_pct > 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for BTST backtesting results"""
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
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # BTST-specific metrics
    gap_up_count: int
    gap_down_count: int
    gap_up_rate: float
    gap_down_rate: float
    avg_gap_up_pct: float
    avg_gap_down_pct: float

    # Activity metrics
    signals_per_week: float

    # Sector breakdown
    sector_performance: Dict[str, Dict]

    # Distribution
    return_distribution: Dict[str, int]


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BTSTBacktestV1:
    """
    BTST backtesting engine v1 for Indian market
    Entry at 3:15 PM, Exit at 9:15 AM next day
    """

    def __init__(self, db_path: str, verbose: bool = False):
        """
        Initialize BTST backtesting engine

        Args:
            db_path: Path to SQLite database with historical data
            verbose: Enable detailed logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self.trades: List[Trade] = []
        self.trade_counter = 0
        self.max_candidates = 10  # Lower than swing

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
        """Get stock metadata"""
        query = """
        SELECT
            symbol,
            sector,
            market_cap,
            table_name
        FROM stock_metadata
        WHERE data_status = 'complete'
        """

        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded metadata for {len(df)} stocks")
        return df

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price and indicator data for a stock"""
        table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"

        query = f"""
        SELECT
            date, open, high, low, close, volume,
            ema_20, ema_50,
            rsi, macd, macd_signal,
            volume_ratio
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

    def apply_sector_filters(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic sector filters for BTST
        BTST doesn't need strict fundamentals, just exclude volatile sectors
        """
        excluded_sectors = ['PHARMA']  # Too volatile for overnight
        filtered = metadata[~metadata['sector'].isin(excluded_sectors)].copy()

        logger.debug(f"Sector filters: {len(metadata)} -> {len(filtered)} stocks")
        return filtered

    def check_btst_criteria(self, symbol: str, current_date: str, hist_df: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check BTST entry criteria at 3:15 PM

        Criteria:
        1. Day Gain: 2.0-3.5% (30 points)
        2. Volume: >= 1.5x (25 points)
        3. High Proximity: >= 90% (20 points)
        4. Trend: Above 20 EMA (15 points)
        5. Momentum: MACD + RSI (10 points)

        Min score: 60 points

        Returns:
            (passes, score, reason)
        """
        if len(hist_df) < 50:
            return False, 0, "Insufficient data"

        latest = hist_df.iloc[-1]
        prev_20 = hist_df.iloc[-21:-1]  # Last 20 days for averages

        score = 0
        reasons = []

        # 1. Day Gain: 2.0-3.5% (30 points)
        day_gain_pct = ((latest['close'] - latest['open']) / latest['open']) * 100

        if 2.0 <= day_gain_pct <= 3.5:
            score += 30
            reasons.append(f"Day gain {day_gain_pct:.1f}%")
        elif day_gain_pct > 3.5:
            score += 15  # Too much, overbought risk
            reasons.append(f"Day gain {day_gain_pct:.1f}% (overbought risk)")
        else:
            return False, 0, f"Day gain {day_gain_pct:.1f}% below 2%"

        # 2. Volume: >= 1.5x (25 points)
        avg_volume = prev_20['volume'].mean()
        volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 0

        if volume_ratio >= 1.5:
            score += 25
            reasons.append(f"Volume {volume_ratio:.1f}x")
        else:
            score += 10
            reasons.append(f"Volume {volume_ratio:.1f}x (below 1.5x)")

        # 3. High Proximity: >= 90% (20 points)
        high_prox = ((latest['close'] - latest['low']) /
                     (latest['high'] - latest['low']) * 100) if latest['high'] > latest['low'] else 0

        if high_prox >= 90:
            score += 20
            reasons.append(f"Near high {high_prox:.0f}%")
        elif high_prox >= 80:
            score += 15
            reasons.append(f"Near high {high_prox:.0f}%")
        else:
            score += 5

        # 4. Trend: Above 20 EMA (15 points)
        ema_20 = latest.get('ema_20', 0)
        if latest['close'] > ema_20:
            score += 15
            reasons.append("Above 20 EMA")
        else:
            score += 5

        # 5. Momentum: MACD/RSI (10 points)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        rsi = latest.get('rsi', 50)

        if macd > macd_signal and rsi > 50:
            score += 10
            reasons.append("Positive momentum")
        else:
            score += 5

        passes = score >= 60
        reason_str = "; ".join(reasons)

        return passes, score, reason_str

    def get_next_trading_day(self, current_date: str) -> str:
        """Get next trading day after current date"""
        current_dt = pd.to_datetime(current_date)
        next_dt = current_dt + pd.Timedelta(days=1)

        # Skip weekends
        while next_dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_dt += pd.Timedelta(days=1)

        return next_dt.strftime('%Y-%m-%d')

    def scan_date(self, scan_date: str, metadata: pd.DataFrame) -> List[Dict]:
        """
        Simulate end-of-day scan at 3:15 PM

        Args:
            scan_date: Date to scan (YYYY-MM-DD)
            metadata: Stock metadata

        Returns:
            List of candidate dictionaries
        """
        candidates = []

        # Apply sector filters
        filtered_metadata = self.apply_sector_filters(metadata)

        for _, stock in filtered_metadata.iterrows():
            symbol = stock['symbol']
            sector = stock['sector']

            # Get historical data up to scan date
            lookback_date = (pd.to_datetime(scan_date) - timedelta(days=100)).strftime('%Y-%m-%d')
            df = self.get_historical_data(symbol, lookback_date, scan_date)

            if df.empty or len(df) < 50:
                continue

            # Check BTST criteria
            passes, score, reason = self.check_btst_criteria(symbol, scan_date, df)

            if not passes:
                continue

            latest = df.iloc[-1]

            # Calculate day gain and high proximity for tracking
            day_gain_pct = ((latest['close'] - latest['open']) / latest['open']) * 100
            high_prox = ((latest['close'] - latest['low']) /
                         (latest['high'] - latest['low']) * 100) if latest['high'] > latest['low'] else 0

            candidates.append({
                'symbol': symbol,
                'sector': sector,
                'score': score,
                'entry_price': latest['close'],
                'day_gain_pct': day_gain_pct,
                'volume_ratio': latest.get('volume_ratio', 0),
                'high_proximity_pct': high_prox,
                'rsi': latest.get('rsi', 0),
                'reason': reason
            })

        # Sort by score and take top 10
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.max_candidates]

    def execute_btst_trade(self, candidate: Dict, entry_date: str) -> Optional[Trade]:
        """
        Execute BTST trade: buy at close, sell at next open

        Args:
            candidate: Candidate dictionary from scan
            entry_date: Entry date (buy at close)

        Returns:
            Trade object or None if exit data unavailable
        """
        symbol = candidate['symbol']
        entry_price = candidate['entry_price']

        # Get next trading day
        exit_date = self.get_next_trading_day(entry_date)

        # Get exit data (open price of next day)
        exit_df = self.get_historical_data(symbol, exit_date, exit_date)

        if exit_df.empty:
            return None

        exit_price = exit_df.iloc[0]['open']

        # Calculate gap
        gap_pct = ((exit_price - entry_price) / entry_price) * 100

        # Calculate P/L
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl_abs = exit_price - entry_price

        self.trade_counter += 1

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            sector=candidate['sector'],
            entry_date=entry_date,
            entry_price=entry_price,
            score=candidate['score'],
            exit_date=exit_date,
            exit_price=exit_price,
            hold_days=1,
            profit_loss_pct=pnl_pct,
            profit_loss_abs=pnl_abs,
            gap_pct=gap_pct,
            day_gain_pct=candidate['day_gain_pct'],
            volume_ratio_entry=candidate['volume_ratio'],
            high_proximity_pct=candidate['high_proximity_pct'],
            rsi_entry=candidate['rsi']
        )

        return trade

    def run_backtest(self, start_date: str, end_date: str) -> List[Trade]:
        """
        Run complete BTST backtest simulation

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            List of all trades executed
        """
        logger.info("=" * 80)
        logger.info(f"Starting BTST v1 Backtest (India): {start_date} to {end_date}")
        logger.info("=" * 80)

        self.connect_db()

        # Get stock metadata
        metadata = self.get_stock_metadata()

        # Generate list of trading dates
        date_range = pd.bdate_range(start=start_date, end=end_date)
        total_dates = len(date_range)

        logger.info(f"Total trading days to simulate: {total_dates}")
        logger.info(f"Stock universe: {len(metadata)} stocks")

        all_trades: List[Trade] = []
        scan_results_count = []

        for i, scan_date in enumerate(date_range):
            scan_date_str = scan_date.strftime('%Y-%m-%d')

            # Progress logging
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Progress: {i+1}/{total_dates} days | Trades: {len(all_trades)}")

            # Run EOD scan at 3:15 PM
            candidates = self.scan_date(scan_date_str, metadata)
            scan_results_count.append(len(candidates))

            if self.verbose and candidates:
                logger.debug(f"Scan {scan_date_str}: {len(candidates)} BTST candidates")

            # Execute BTST trades
            for candidate in candidates:
                trade = self.execute_btst_trade(candidate, scan_date_str)

                if trade:
                    all_trades.append(trade)

                    if self.verbose:
                        logger.debug(
                            f"BTST: {trade.symbol} | Entry: ₹{trade.entry_price:.2f} @ {trade.entry_date} | "
                            f"Exit: ₹{trade.exit_price:.2f} @ {trade.exit_date} | "
                            f"P/L: {trade.profit_loss_pct:+.2f}% | Gap: {trade.gap_pct:+.2f}%"
                        )

        self.trades = all_trades

        logger.info("=" * 80)
        logger.info(f"Backtest Complete")
        logger.info(f"Total BTST trades: {len(all_trades)}")
        logger.info(f"Average candidates per scan: {np.mean(scan_results_count):.1f}")
        logger.info("=" * 80)

        self.close_db()
        return all_trades

    def calculate_performance_metrics(self, trades: List[Trade],
                                     start_date: str, end_date: str) -> PerformanceMetrics:
        """Calculate comprehensive BTST performance metrics"""
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

        # Max drawdown
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

        # BTST-specific: Gap analysis
        gap_ups = [t for t in trades if t.gap_pct > 0]
        gap_downs = [t for t in trades if t.gap_pct < 0]

        gap_up_count = len(gap_ups)
        gap_down_count = len(gap_downs)
        gap_up_rate = (gap_up_count / total_trades * 100) if total_trades > 0 else 0
        gap_down_rate = (gap_down_count / total_trades * 100) if total_trades > 0 else 0

        avg_gap_up_pct = np.mean([t.gap_pct for t in gap_ups]) if gap_ups else 0
        avg_gap_down_pct = np.mean([t.gap_pct for t in gap_downs]) if gap_downs else 0

        # Activity metrics
        weeks = total_days / 7
        signals_per_week = total_trades / weeks if weeks > 0 else 0

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
            '< -5%': sum(1 for t in trades if t.profit_loss_pct < -5),
            '-5% to -2%': sum(1 for t in trades if -5 <= t.profit_loss_pct < -2),
            '-2% to 0%': sum(1 for t in trades if -2 <= t.profit_loss_pct < 0),
            '0% to 2%': sum(1 for t in trades if 0 <= t.profit_loss_pct < 2),
            '2% to 5%': sum(1 for t in trades if 2 <= t.profit_loss_pct < 5),
            '> 5%': sum(1 for t in trades if t.profit_loss_pct >= 5)
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
            gap_up_count=gap_up_count,
            gap_down_count=gap_down_count,
            gap_up_rate=gap_up_rate,
            gap_down_rate=gap_down_rate,
            avg_gap_up_pct=avg_gap_up_pct,
            avg_gap_down_pct=avg_gap_down_pct,
            signals_per_week=signals_per_week,
            sector_performance=sector_performance,
            return_distribution=return_distribution
        )

        return metrics

    def save_results(self, trades: List[Trade], metrics: PerformanceMetrics,
                    output_dir: str):
        """Save BTST backtest results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Save trades CSV
        trades_file = output_path / f'btst_backtest_v1_trades_{timestamp}.csv'
        trades_df = pd.DataFrame([asdict(t) for t in trades])
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved detailed trades: {trades_file}")

        # 2. Save metrics JSON
        metrics_file = output_path / f'btst_backtest_v1_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved performance metrics: {metrics_file}")

        # 3. Save report
        report_file = output_path / f'btst_backtest_v1_report_{timestamp}.txt'
        report = self.generate_report(metrics)
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved backtest report: {report_file}")

        return trades_file, metrics_file, report_file

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate human-readable BTST performance report"""
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS - BTST STRATEGY V1 (INDIA - NIFTY 500)")
        report.append("Buy Today Sell Tomorrow - Entry 3:15 PM, Exit 9:15 AM")
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
        report.append("")

        # BTST-Specific: Gap Analysis
        report.append("GAP ANALYSIS (OVERNIGHT)")
        report.append("-" * 80)
        report.append(f"Gap Up:           {metrics.gap_up_count} trades ({metrics.gap_up_rate:.1f}%)")
        report.append(f"Avg Gap Up:       +{metrics.avg_gap_up_pct:.2f}%")
        report.append(f"Gap Down:         {metrics.gap_down_count} trades ({metrics.gap_down_rate:.1f}%)")
        report.append(f"Avg Gap Down:     {metrics.avg_gap_down_pct:.2f}%")
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
        report.append("BTST V1 STRATEGY:")
        report.append("- Entry: 3:15 PM (market close)")
        report.append("- Exit: 9:15 AM next day (market open)")
        report.append("- Hold: Always 1 day")
        report.append("- Criteria: Day gain 2-3.5%, Volume 1.5x+, High prox 90%+")
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
        description='Backtest BTST v1 Strategy for Indian Market (Nifty 500)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for last 6 months
  python btst_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01

  # With verbose logging
  python btst_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01 --verbose
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
        help='Path to SQLite database'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/backtesting/results',
        help='Output directory for results'
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

    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        return

    # Initialize backtester
    backtester = BTSTBacktestV1(
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

        # Print summary
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
