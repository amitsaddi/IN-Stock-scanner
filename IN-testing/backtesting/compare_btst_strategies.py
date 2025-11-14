"""
Comprehensive Swing Strategy Comparison Module for v1 vs v2 Backtesting Results (India)

This module provides extensive analysis and reporting capabilities to compare
the performance of v1 (current) vs v2 (enhanced) BTST (Buy Today Sell Tomorrow) strategies.

Features:
- Side-by-side performance comparison
- Entry signal analysis (including v2's 3 new signals)
- Sector performance breakdown
- Risk-adjusted returns comparison
- Statistical significance testing
- Filter impact analysis (v2 specific)
- Multiple output formats (HTML, TXT, CSV)
- Automated recommendations

Author: Strategy Analysis Framework
Date: 2025-11-11
Version: 1.0
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "IN-testing" / "scripts"))

from db_schema import NiftyDatabaseManager

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
class ComparisonMetrics:
    """Comprehensive comparison metrics between v1 and v2"""
    # Overall performance deltas
    total_trades_v1: int
    total_trades_v2: int
    trades_diff_pct: float

    win_rate_v1: float
    win_rate_v2: float
    win_rate_diff_pct: float

    avg_return_v1: float
    avg_return_v2: float
    avg_return_diff_pct: float

    total_return_v1: float
    total_return_v2: float
    total_return_diff_pct: float

    profit_factor_v1: float
    profit_factor_v2: float
    profit_factor_diff_pct: float

    sharpe_ratio_v1: float
    sharpe_ratio_v2: float
    sharpe_ratio_diff_pct: float

    max_drawdown_v1: float
    max_drawdown_v2: float
    max_drawdown_diff_pct: float

    signals_per_week_v1: float
    signals_per_week_v2: float
    signals_per_week_diff_pct: float

    # Statistical significance
    returns_ttest_pvalue: float
    returns_statistically_significant: bool
    win_rate_chi2_pvalue: float
    win_rate_statistically_significant: bool

    # Recommendation
    recommended_strategy: str
    recommendation_confidence: str
    recommendation_reasons: List[str]


# ============================================================================
# COMPARISON ENGINE
# ============================================================================

class StrategyComparison:
    """
    Comprehensive comparison engine for v1 vs v2 strategies
    """

    def __init__(self, v1_metrics_path: str, v1_trades_path: str,
                 v2_metrics_path: str, v2_trades_path: str):
        """
        Initialize comparison engine

        Args:
            v1_metrics_path: Path to v1 metrics JSON
            v1_trades_path: Path to v1 trades CSV
            v2_metrics_path: Path to v2 metrics JSON
            v2_trades_path: Path to v2 trades CSV
        """
        self.v1_metrics_path = v1_metrics_path
        self.v1_trades_path = v1_trades_path
        self.v2_metrics_path = v2_metrics_path
        self.v2_trades_path = v2_trades_path

        self.v1_metrics = None
        self.v1_trades = None
        self.v2_metrics = None
        self.v2_trades = None

        self.comparison = None

    def load_backtest_results(self):
        """Load all backtest results from files"""
        logger.info("Loading backtest results...")

        try:
            # Load v1 metrics
            with open(self.v1_metrics_path, 'r') as f:
                self.v1_metrics = json.load(f)
            logger.info(f"Loaded v1 metrics: {self.v1_metrics_path}")

            # Load v1 trades
            self.v1_trades = pd.read_csv(self.v1_trades_path)
            logger.info(f"Loaded v1 trades: {len(self.v1_trades)} trades")

            # Load v2 metrics
            with open(self.v2_metrics_path, 'r') as f:
                self.v2_metrics = json.load(f)
            logger.info(f"Loaded v2 metrics: {self.v2_metrics_path}")

            # Load v2 trades
            self.v2_trades = pd.read_csv(self.v2_trades_path)
            logger.info(f"Loaded v2 trades: {len(self.v2_trades)} trades")

            logger.info("All results loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {e}")
            raise

    def calculate_percentage_diff(self, v1_val: float, v2_val: float) -> float:
        """Calculate percentage difference (v2 vs v1)"""
        if v1_val == 0:
            return 0 if v2_val == 0 else 100
        return ((v2_val - v1_val) / abs(v1_val)) * 100

    def compare_metrics(self) -> ComparisonMetrics:
        """
        Compare overall performance metrics

        Returns:
            ComparisonMetrics object with comprehensive comparison
        """
        logger.info("Comparing performance metrics...")

        v1 = self.v1_metrics
        v2 = self.v2_metrics

        # Overall performance
        total_trades_v1 = v1['total_trades']
        total_trades_v2 = v2['total_trades']
        trades_diff_pct = self.calculate_percentage_diff(total_trades_v1, total_trades_v2)

        win_rate_v1 = v1['win_rate']
        win_rate_v2 = v2['win_rate']
        win_rate_diff_pct = self.calculate_percentage_diff(win_rate_v1, win_rate_v2)

        avg_return_v1 = v1['avg_trade_pct']
        avg_return_v2 = v2['avg_trade_pct']
        avg_return_diff_pct = self.calculate_percentage_diff(avg_return_v1, avg_return_v2)

        total_return_v1 = v1['total_return_pct']
        total_return_v2 = v2['total_return_pct']
        total_return_diff_pct = self.calculate_percentage_diff(total_return_v1, total_return_v2)

        profit_factor_v1 = v1['profit_factor']
        profit_factor_v2 = v2['profit_factor']
        profit_factor_diff_pct = self.calculate_percentage_diff(profit_factor_v1, profit_factor_v2)

        sharpe_ratio_v1 = v1['sharpe_ratio']
        sharpe_ratio_v2 = v2['sharpe_ratio']
        sharpe_ratio_diff_pct = self.calculate_percentage_diff(sharpe_ratio_v1, sharpe_ratio_v2)

        max_drawdown_v1 = v1['max_drawdown_pct']
        max_drawdown_v2 = v2['max_drawdown_pct']
        max_drawdown_diff_pct = self.calculate_percentage_diff(max_drawdown_v1, max_drawdown_v2)

        signals_per_week_v1 = v1['signals_per_week']
        signals_per_week_v2 = v2['signals_per_week']
        signals_per_week_diff_pct = self.calculate_percentage_diff(signals_per_week_v1, signals_per_week_v2)

        # Statistical significance tests
        returns_ttest_pvalue, returns_sig = self.test_returns_significance()
        win_rate_chi2_pvalue, win_rate_sig = self.test_win_rate_significance()

        # Generate recommendation
        recommended_strategy, confidence, reasons = self.generate_recommendations()

        comparison = ComparisonMetrics(
            total_trades_v1=total_trades_v1,
            total_trades_v2=total_trades_v2,
            trades_diff_pct=trades_diff_pct,
            win_rate_v1=win_rate_v1,
            win_rate_v2=win_rate_v2,
            win_rate_diff_pct=win_rate_diff_pct,
            avg_return_v1=avg_return_v1,
            avg_return_v2=avg_return_v2,
            avg_return_diff_pct=avg_return_diff_pct,
            total_return_v1=total_return_v1,
            total_return_v2=total_return_v2,
            total_return_diff_pct=total_return_diff_pct,
            profit_factor_v1=profit_factor_v1,
            profit_factor_v2=profit_factor_v2,
            profit_factor_diff_pct=profit_factor_diff_pct,
            sharpe_ratio_v1=sharpe_ratio_v1,
            sharpe_ratio_v2=sharpe_ratio_v2,
            sharpe_ratio_diff_pct=sharpe_ratio_diff_pct,
            max_drawdown_v1=max_drawdown_v1,
            max_drawdown_v2=max_drawdown_v2,
            max_drawdown_diff_pct=max_drawdown_diff_pct,
            signals_per_week_v1=signals_per_week_v1,
            signals_per_week_v2=signals_per_week_v2,
            signals_per_week_diff_pct=signals_per_week_diff_pct,
            returns_ttest_pvalue=returns_ttest_pvalue,
            returns_statistically_significant=returns_sig,
            win_rate_chi2_pvalue=win_rate_chi2_pvalue,
            win_rate_statistically_significant=win_rate_sig,
            recommended_strategy=recommended_strategy,
            recommendation_confidence=confidence,
            recommendation_reasons=reasons
        )

        self.comparison = comparison
        logger.info("Metrics comparison complete")
        return comparison

    def test_returns_significance(self) -> Tuple[float, bool]:
        """
        Test if returns are statistically significantly different (t-test)

        Returns:
            (p-value, is_significant)
        """
        v1_returns = self.v1_trades['profit_loss_pct'].values
        v2_returns = self.v2_trades['profit_loss_pct'].values

        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(v1_returns, v2_returns, equal_var=False)

        # Significant if p < 0.05
        is_significant = p_value < 0.05

        logger.info(f"Returns t-test: t={t_stat:.3f}, p={p_value:.4f}, sig={is_significant}")
        return p_value, is_significant

    def test_win_rate_significance(self) -> Tuple[float, bool]:
        """
        Test if win rates are statistically significantly different (chi-square)

        Returns:
            (p-value, is_significant)
        """
        v1_wins = self.v1_metrics['winning_trades']
        v1_losses = self.v1_metrics['losing_trades']
        v2_wins = self.v2_metrics['winning_trades']
        v2_losses = self.v2_metrics['losing_trades']

        # Contingency table
        observed = np.array([[v1_wins, v1_losses],
                            [v2_wins, v2_losses]])

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        # Significant if p < 0.05
        is_significant = p_value < 0.05

        logger.info(f"Win rate chi-square: chi2={chi2:.3f}, p={p_value:.4f}, sig={is_significant}")
        return p_value, is_significant

    def generate_recommendations(self) -> Tuple[str, str, List[str]]:
        """
        Generate strategy recommendation based on comparison

        v2 is recommended if:
        - 30%+ more signals
        - Win rate within 5% of v1
        - Average return 10%+ higher
        - Max drawdown equal or better
        - Better Sharpe ratio

        Returns:
            (recommended_strategy, confidence_level, reasons)
        """
        v1 = self.v1_metrics
        v2 = self.v2_metrics

        reasons = []
        v2_score = 0
        max_score = 5

        # Criterion 1: Signal volume (30%+ more)
        signal_increase_pct = self.calculate_percentage_diff(
            v1['signals_per_week'], v2['signals_per_week']
        )
        if signal_increase_pct >= 30:
            v2_score += 1
            reasons.append(f"v2 generates {signal_increase_pct:.1f}% more signals ({v2['signals_per_week']:.1f} vs {v1['signals_per_week']:.1f} per week)")
        else:
            reasons.append(f"v2 signal increase ({signal_increase_pct:.1f}%) below 30% threshold")

        # Criterion 2: Win rate within 5%
        win_rate_diff = v2['win_rate'] - v1['win_rate']
        if abs(win_rate_diff) <= 5:
            v2_score += 1
            reasons.append(f"v2 win rate comparable ({v2['win_rate']:.1f}% vs {v1['win_rate']:.1f}%, diff: {win_rate_diff:+.1f}%)")
        else:
            if win_rate_diff > 0:
                reasons.append(f"v2 win rate significantly higher ({win_rate_diff:+.1f}%)")
            else:
                reasons.append(f"v2 win rate lower by {abs(win_rate_diff):.1f}% (concern)")

        # Criterion 3: Average return 10%+ higher
        avg_return_increase_pct = self.calculate_percentage_diff(
            v1['avg_trade_pct'], v2['avg_trade_pct']
        )
        if avg_return_increase_pct >= 10:
            v2_score += 1
            reasons.append(f"v2 average return {avg_return_increase_pct:.1f}% higher ({v2['avg_trade_pct']:.2f}% vs {v1['avg_trade_pct']:.2f}%)")
        else:
            if avg_return_increase_pct >= 0:
                reasons.append(f"v2 average return increase ({avg_return_increase_pct:.1f}%) below 10% threshold")
            else:
                reasons.append(f"v2 average return lower by {abs(avg_return_increase_pct):.1f}% (concern)")

        # Criterion 4: Max drawdown equal or better
        drawdown_change_pct = self.calculate_percentage_diff(
            v1['max_drawdown_pct'], v2['max_drawdown_pct']
        )
        if v2['max_drawdown_pct'] <= v1['max_drawdown_pct']:
            v2_score += 1
            reasons.append(f"v2 max drawdown equal or better ({v2['max_drawdown_pct']:.2f}% vs {v1['max_drawdown_pct']:.2f}%)")
        else:
            reasons.append(f"v2 max drawdown worse by {abs(drawdown_change_pct):.1f}% (concern)")

        # Criterion 5: Better Sharpe ratio
        if v2['sharpe_ratio'] > v1['sharpe_ratio']:
            v2_score += 1
            sharpe_diff_pct = self.calculate_percentage_diff(v1['sharpe_ratio'], v2['sharpe_ratio'])
            reasons.append(f"v2 Sharpe ratio better ({v2['sharpe_ratio']:.2f} vs {v1['sharpe_ratio']:.2f}, +{sharpe_diff_pct:.1f}%)")
        else:
            sharpe_diff_pct = self.calculate_percentage_diff(v1['sharpe_ratio'], v2['sharpe_ratio'])
            reasons.append(f"v2 Sharpe ratio lower ({sharpe_diff_pct:.1f}%)")

        # Determine recommendation
        if v2_score >= 4:
            recommended = "v2"
            confidence = "HIGH"
        elif v2_score >= 3:
            recommended = "v2"
            confidence = "MEDIUM"
        elif v2_score >= 2:
            recommended = "NEUTRAL"
            confidence = "LOW"
        else:
            recommended = "v1"
            confidence = "MEDIUM"

        logger.info(f"Recommendation: {recommended} ({confidence} confidence) - Score: {v2_score}/{max_score}")
        return recommended, confidence, reasons

    def compare_entry_signals(self) -> pd.DataFrame:
        """
        Compare entry signal performance between v1 and v2

        Returns:
            DataFrame with side-by-side signal comparison
        """
        logger.info("Comparing entry signals...")

        v1_signals = self.v1_metrics.get('entry_type_performance', {})
        v2_signals = self.v2_metrics.get('entry_type_performance', {})

        # Get all unique signals
        all_signals = set(list(v1_signals.keys()) + list(v2_signals.keys()))

        comparison_data = []

        for signal in sorted(all_signals):
            v1_data = v1_signals.get(signal, {'count': 0, 'win_rate': 0, 'avg_return': 0})
            v2_data = v2_signals.get(signal, {'count': 0, 'win_rate': 0, 'avg_return': 0})

            is_new_signal = signal in ['bollinger_bounce', 'volume_surge', 'consolidation_breakout']

            comparison_data.append({
                'signal': signal.replace('_', ' ').title(),
                'is_new_v2': is_new_signal,
                'v1_count': v1_data['count'],
                'v2_count': v2_data['count'],
                'count_diff_pct': self.calculate_percentage_diff(v1_data['count'], v2_data['count']),
                'v1_win_rate': v1_data['win_rate'],
                'v2_win_rate': v2_data['win_rate'],
                'win_rate_diff': v2_data['win_rate'] - v1_data['win_rate'],
                'v1_avg_return': v1_data['avg_return'],
                'v2_avg_return': v2_data['avg_return'],
                'avg_return_diff': v2_data['avg_return'] - v1_data['avg_return']
            })

        df = pd.DataFrame(comparison_data)
        logger.info(f"Entry signal comparison complete: {len(df)} signals")
        return df

    def compare_sectors(self) -> pd.DataFrame:
        """
        Compare sector performance between v1 and v2

        Returns:
            DataFrame with side-by-side sector comparison
        """
        logger.info("Comparing sector performance...")

        v1_sectors = self.v1_metrics.get('sector_performance', {})
        v2_sectors = self.v2_metrics.get('sector_performance', {})

        # Get all unique sectors
        all_sectors = set(list(v1_sectors.keys()) + list(v2_sectors.keys()))

        comparison_data = []

        for sector in sorted(all_sectors):
            v1_data = v1_sectors.get(sector, {'count': 0, 'win_rate': 0, 'avg_return': 0})
            v2_data = v2_sectors.get(sector, {'count': 0, 'win_rate': 0, 'avg_return': 0})

            comparison_data.append({
                'sector': sector,
                'v1_count': v1_data['count'],
                'v2_count': v2_data['count'],
                'count_diff_pct': self.calculate_percentage_diff(v1_data['count'], v2_data['count']),
                'v1_win_rate': v1_data['win_rate'],
                'v2_win_rate': v2_data['win_rate'],
                'win_rate_diff': v2_data['win_rate'] - v1_data['win_rate'],
                'v1_avg_return': v1_data['avg_return'],
                'v2_avg_return': v2_data['avg_return'],
                'avg_return_diff': v2_data['avg_return'] - v1_data['avg_return']
            })

        df = pd.DataFrame(comparison_data)
        logger.info(f"Sector comparison complete: {len(df)} sectors")
        return df

    def analyze_risk_adjusted_returns(self) -> Dict:
        """
        Calculate and compare risk-adjusted return metrics

        Returns:
            Dictionary with risk-adjusted metrics
        """
        logger.info("Analyzing risk-adjusted returns...")

        v1 = self.v1_metrics
        v2 = self.v2_metrics

        # Return per unit of risk (avg_return / max_drawdown)
        v1_return_risk_ratio = abs(v1['avg_trade_pct'] / v1['max_drawdown_pct']) if v1['max_drawdown_pct'] > 0 else 0
        v2_return_risk_ratio = abs(v2['avg_trade_pct'] / v2['max_drawdown_pct']) if v2['max_drawdown_pct'] > 0 else 0

        # Calmar ratio (total_return / max_drawdown)
        v1_calmar = abs(v1['total_return_pct'] / v1['max_drawdown_pct']) if v1['max_drawdown_pct'] > 0 else 0
        v2_calmar = abs(v2['total_return_pct'] / v2['max_drawdown_pct']) if v2['max_drawdown_pct'] > 0 else 0

        # Sortino ratio approximation (using only downside deviation)
        v1_losses = [t for t in self.v1_trades['profit_loss_pct'].values if t < 0]
        v2_losses = [t for t in self.v2_trades['profit_loss_pct'].values if t < 0]

        v1_downside_std = np.std(v1_losses) if len(v1_losses) > 1 else 0
        v2_downside_std = np.std(v2_losses) if len(v2_losses) > 1 else 0

        v1_sortino = (v1['avg_trade_pct'] / v1_downside_std) if v1_downside_std > 0 else 0
        v2_sortino = (v2['avg_trade_pct'] / v2_downside_std) if v2_downside_std > 0 else 0

        risk_metrics = {
            'v1_return_risk_ratio': v1_return_risk_ratio,
            'v2_return_risk_ratio': v2_return_risk_ratio,
            'return_risk_improvement_pct': self.calculate_percentage_diff(v1_return_risk_ratio, v2_return_risk_ratio),
            'v1_sharpe_ratio': v1['sharpe_ratio'],
            'v2_sharpe_ratio': v2['sharpe_ratio'],
            'sharpe_improvement_pct': self.calculate_percentage_diff(v1['sharpe_ratio'], v2['sharpe_ratio']),
            'v1_sortino_ratio': v1_sortino,
            'v2_sortino_ratio': v2_sortino,
            'sortino_improvement_pct': self.calculate_percentage_diff(v1_sortino, v2_sortino),
            'v1_calmar_ratio': v1_calmar,
            'v2_calmar_ratio': v2_calmar,
            'calmar_improvement_pct': self.calculate_percentage_diff(v1_calmar, v2_calmar)
        }

        logger.info("Risk-adjusted returns analysis complete")
        return risk_metrics

    def analyze_trade_distribution(self) -> Dict:
        """
        Analyze return distribution and hold period

        Returns:
            Dictionary with distribution metrics
        """
        logger.info("Analyzing trade distributions...")

        v1 = self.v1_metrics
        v2 = self.v2_metrics

        # Return distributions
        v1_dist = v1.get('return_distribution', {})
        v2_dist = v2.get('return_distribution', {})

        # Hold period analysis
        v1_hold = self.v1_trades['hold_days'].values
        v2_hold = self.v2_trades['hold_days'].values

        # Win/loss streaks
        v1_returns = self.v1_trades['profit_loss_pct'].values
        v2_returns = self.v2_trades['profit_loss_pct'].values

        v1_max_win_streak = self._calculate_max_streak(v1_returns, positive=True)
        v1_max_loss_streak = self._calculate_max_streak(v1_returns, positive=False)
        v2_max_win_streak = self._calculate_max_streak(v2_returns, positive=True)
        v2_max_loss_streak = self._calculate_max_streak(v2_returns, positive=False)

        distribution = {
            'return_distribution': {
                'v1': v1_dist,
                'v2': v2_dist
            },
            'hold_period': {
                # BTST is always 1 day hold
                'v1_avg': v1.get('avg_hold_days', 1.0),
                'v2_avg': v2.get('avg_hold_days', 1.0),
                'v1_median': v1.get('median_hold_days', 1.0),
                'v2_median': v2.get('median_hold_days', 1.0),
                'v1_std': np.std(v1_hold) if len(v1_hold) > 0 else 0.0,
                'v2_std': np.std(v2_hold) if len(v2_hold) > 0 else 0.0
            },
            'streaks': {
                'v1_max_win_streak': v1_max_win_streak,
                'v2_max_win_streak': v2_max_win_streak,
                'v1_max_loss_streak': v1_max_loss_streak,
                'v2_max_loss_streak': v2_max_loss_streak
            }
        }

        logger.info("Trade distribution analysis complete")
        return distribution

    def _calculate_max_streak(self, returns: np.ndarray, positive: bool = True) -> int:
        """Calculate maximum winning or losing streak"""
        max_streak = 0
        current_streak = 0

        for ret in returns:
            if (positive and ret > 0) or (not positive and ret < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def analyze_v2_filter_impact(self) -> Dict:
        """
        Analyze impact of v2-specific filters (ATR, Liquidity)

        Returns:
            Dictionary with filter impact metrics
        """
        logger.info("Analyzing v2 filter impact...")

        v2 = self.v2_metrics

        # ATR filter impact
        atr_impact = v2.get('atr_filter_impact', {})

        # Liquidity filter impact
        liquidity_impact = v2.get('liquidity_filter_impact', {})

        # New signal performance
        new_signals = v2.get('new_signal_performance', {})

        # Calculate how many MORE candidates v2 captured
        v1_total = self.v1_metrics['total_trades']
        v2_total = self.v2_metrics['total_trades']
        additional_candidates = v2_total - v1_total
        additional_candidates_pct = (additional_candidates / v1_total * 100) if v1_total > 0 else 0

        # RSI range impact (30-70 vs 35-65)
        v1_rsi_trades = self.v1_trades
        v2_rsi_trades = self.v2_trades

        # Count trades in the wider ranges (30-35 and 65-70)
        v2_wider_range_count = len(v2_rsi_trades[
            ((v2_rsi_trades['rsi_entry'] >= 30) & (v2_rsi_trades['rsi_entry'] < 35)) |
            ((v2_rsi_trades['rsi_entry'] > 65) & (v2_rsi_trades['rsi_entry'] <= 70))
        ]) if 'rsi_entry' in v2_rsi_trades.columns else 0

        filter_impact = {
            'additional_candidates': additional_candidates,
            'additional_candidates_pct': additional_candidates_pct,
            'atr_filter': atr_impact,
            'liquidity_filter': liquidity_impact,
            'new_signals_performance': new_signals,
            'wider_rsi_range_trades': v2_wider_range_count,
            'wider_rsi_range_pct': (v2_wider_range_count / v2_total * 100) if v2_total > 0 else 0
        }

        logger.info("v2 filter impact analysis complete")
        return filter_impact

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_html_report(self, output_path: str):
        """
        Generate comprehensive HTML report with charts and tables

        Args:
            output_path: Path to save HTML file
        """
        logger.info("Generating HTML report...")

        # Get all comparison data
        signals_df = self.compare_entry_signals()
        sectors_df = self.compare_sectors()
        risk_metrics = self.analyze_risk_adjusted_returns()
        distribution = self.analyze_trade_distribution()
        filter_impact = self.analyze_v2_filter_impact()

        html_content = self._build_html_report(
            signals_df, sectors_df, risk_metrics, distribution, filter_impact
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {output_path}")

    def _build_html_report(self, signals_df, sectors_df, risk_metrics,
                          distribution, filter_impact) -> str:
        """Build HTML report content"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTST Strategy Comparison: v1 vs v2 (India)</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }}
        .metric-diff {{
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 5px;
        }}
        .metric-diff.positive {{
            color: #28a745;
        }}
        .metric-diff.negative {{
            color: #dc3545;
        }}
        .metric-diff.neutral {{
            color: #6c757d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .recommendation h2 {{
            color: white;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            margin-top: 0;
        }}
        .recommendation ul {{
            list-style-type: none;
            padding: 0;
        }}
        .recommendation li {{
            padding: 8px 0 8px 25px;
            position: relative;
        }}
        .recommendation li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge.new {{
            background: #28a745;
            color: white;
        }}
        .badge.high {{
            background: #28a745;
            color: white;
        }}
        .badge.medium {{
            background: #ffc107;
            color: #333;
        }}
        .badge.low {{
            background: #dc3545;
            color: white;
        }}
        .highlight-green {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .highlight-red {{
            background-color: #f8d7da;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Strategy Comparison Report</h1>
        <p>v1 (Current) vs v2 (Enhanced) BTST Strategy</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    {self._build_executive_summary_html()}

    {self._build_overall_performance_html()}

    {self._build_entry_signals_html(signals_df)}

    {self._build_sector_performance_html(sectors_df)}

    {self._build_risk_metrics_html(risk_metrics)}

    {self._build_filter_impact_html(filter_impact)}

    {self._build_distribution_html(distribution)}

    {self._build_statistical_significance_html()}

    {self._build_recommendation_html()}

    <div class="footer">
        <p>Nifty 500 BTST Strategy Analysis Framework</p>
        <p>Backtest Period: {self.v1_metrics['start_date']} to {self.v1_metrics['end_date']}</p>
    </div>
</body>
</html>"""

        return html

    def _build_executive_summary_html(self) -> str:
        """Build executive summary section"""
        comp = self.comparison

        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Total Trades</h4>
                <div class="metric-value">{comp.total_trades_v2}</div>
                <div class="metric-diff {'positive' if comp.trades_diff_pct > 0 else 'negative'}">
                    {comp.trades_diff_pct:+.1f}% vs v1 ({comp.total_trades_v1})
                </div>
            </div>
            <div class="metric-card">
                <h4>Win Rate</h4>
                <div class="metric-value">{comp.win_rate_v2:.1f}%</div>
                <div class="metric-diff {'positive' if comp.win_rate_diff_pct > 0 else 'negative'}">
                    {comp.win_rate_diff_pct:+.1f}% vs v1 ({comp.win_rate_v1:.1f}%)
                </div>
            </div>
            <div class="metric-card">
                <h4>Avg Return</h4>
                <div class="metric-value">{comp.avg_return_v2:+.2f}%</div>
                <div class="metric-diff {'positive' if comp.avg_return_diff_pct > 0 else 'negative'}">
                    {comp.avg_return_diff_pct:+.1f}% vs v1 ({comp.avg_return_v1:+.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <h4>Sharpe Ratio</h4>
                <div class="metric-value">{comp.sharpe_ratio_v2:.2f}</div>
                <div class="metric-diff {'positive' if comp.sharpe_ratio_diff_pct > 0 else 'negative'}">
                    {comp.sharpe_ratio_diff_pct:+.1f}% vs v1 ({comp.sharpe_ratio_v1:.2f})
                </div>
            </div>
        </div>
    </div>
"""

    def _build_overall_performance_html(self) -> str:
        """Build overall performance comparison section"""
        comp = self.comparison

        return f"""
    <div class="section">
        <h2>Overall Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>v1 (Current)</th>
                    <th>v2 (Enhanced)</th>
                    <th>Difference</th>
                    <th>% Change</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Total Trades</strong></td>
                    <td>{comp.total_trades_v1}</td>
                    <td>{comp.total_trades_v2}</td>
                    <td>{comp.total_trades_v2 - comp.total_trades_v1:+d}</td>
                    <td class="{'highlight-green' if comp.trades_diff_pct > 0 else 'highlight-red'}">
                        {comp.trades_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Win Rate</strong></td>
                    <td>{comp.win_rate_v1:.2f}%</td>
                    <td>{comp.win_rate_v2:.2f}%</td>
                    <td>{comp.win_rate_v2 - comp.win_rate_v1:+.2f}%</td>
                    <td class="{'highlight-green' if comp.win_rate_diff_pct > 0 else 'highlight-red'}">
                        {comp.win_rate_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Average Return</strong></td>
                    <td>{comp.avg_return_v1:+.2f}%</td>
                    <td>{comp.avg_return_v2:+.2f}%</td>
                    <td>{comp.avg_return_v2 - comp.avg_return_v1:+.2f}%</td>
                    <td class="{'highlight-green' if comp.avg_return_diff_pct > 0 else 'highlight-red'}">
                        {comp.avg_return_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Total Return</strong></td>
                    <td>{comp.total_return_v1:+.2f}%</td>
                    <td>{comp.total_return_v2:+.2f}%</td>
                    <td>{comp.total_return_v2 - comp.total_return_v1:+.2f}%</td>
                    <td class="{'highlight-green' if comp.total_return_diff_pct > 0 else 'highlight-red'}">
                        {comp.total_return_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Profit Factor</strong></td>
                    <td>{comp.profit_factor_v1:.2f}</td>
                    <td>{comp.profit_factor_v2:.2f}</td>
                    <td>{comp.profit_factor_v2 - comp.profit_factor_v1:+.2f}</td>
                    <td class="{'highlight-green' if comp.profit_factor_diff_pct > 0 else 'highlight-red'}">
                        {comp.profit_factor_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Sharpe Ratio</strong></td>
                    <td>{comp.sharpe_ratio_v1:.2f}</td>
                    <td>{comp.sharpe_ratio_v2:.2f}</td>
                    <td>{comp.sharpe_ratio_v2 - comp.sharpe_ratio_v1:+.2f}</td>
                    <td class="{'highlight-green' if comp.sharpe_ratio_diff_pct > 0 else 'highlight-red'}">
                        {comp.sharpe_ratio_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Max Drawdown</strong></td>
                    <td>{comp.max_drawdown_v1:.2f}%</td>
                    <td>{comp.max_drawdown_v2:.2f}%</td>
                    <td>{comp.max_drawdown_v2 - comp.max_drawdown_v1:+.2f}%</td>
                    <td class="{'highlight-green' if comp.max_drawdown_diff_pct < 0 else 'highlight-red'}">
                        {comp.max_drawdown_diff_pct:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td><strong>Signals/Week</strong></td>
                    <td>{comp.signals_per_week_v1:.2f}</td>
                    <td>{comp.signals_per_week_v2:.2f}</td>
                    <td>{comp.signals_per_week_v2 - comp.signals_per_week_v1:+.2f}</td>
                    <td class="{'highlight-green' if comp.signals_per_week_diff_pct > 0 else 'highlight-red'}">
                        {comp.signals_per_week_diff_pct:+.1f}%
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
"""

    def _build_entry_signals_html(self, signals_df: pd.DataFrame) -> str:
        """Build entry signals comparison section"""
        rows_html = ""
        for _, row in signals_df.iterrows():
            new_badge = '<span class="badge new">NEW v2</span>' if row['is_new_v2'] else ''
            rows_html += f"""
                <tr>
                    <td>{row['signal']}{new_badge}</td>
                    <td>{row['v1_count']}</td>
                    <td>{row['v2_count']}</td>
                    <td class="{'highlight-green' if row['count_diff_pct'] > 0 else 'highlight-red'}">
                        {row['count_diff_pct']:+.1f}%
                    </td>
                    <td>{row['v1_win_rate']:.1f}%</td>
                    <td>{row['v2_win_rate']:.1f}%</td>
                    <td class="{'highlight-green' if row['win_rate_diff'] > 0 else 'highlight-red'}">
                        {row['win_rate_diff']:+.1f}%
                    </td>
                    <td>{row['v1_avg_return']:+.2f}%</td>
                    <td>{row['v2_avg_return']:+.2f}%</td>
                    <td class="{'highlight-green' if row['avg_return_diff'] > 0 else 'highlight-red'}">
                        {row['avg_return_diff']:+.2f}%
                    </td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Entry Signal Analysis</h2>
        <p>Comparison of all entry signals including v2's 3 new signals: Bollinger Bounce, Volume Surge, and Consolidation Breakout.</p>
        <table>
            <thead>
                <tr>
                    <th>Entry Signal</th>
                    <th>v1 Count</th>
                    <th>v2 Count</th>
                    <th>Count Δ%</th>
                    <th>v1 Win Rate</th>
                    <th>v2 Win Rate</th>
                    <th>Win Rate Δ</th>
                    <th>v1 Avg Return</th>
                    <th>v2 Avg Return</th>
                    <th>Return Δ</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
"""

    def _build_sector_performance_html(self, sectors_df: pd.DataFrame) -> str:
        """Build sector performance comparison section"""
        rows_html = ""
        for _, row in sectors_df.iterrows():
            rows_html += f"""
                <tr>
                    <td>{row['sector']}</td>
                    <td>{row['v1_count']}</td>
                    <td>{row['v2_count']}</td>
                    <td class="{'highlight-green' if row['count_diff_pct'] > 0 else 'highlight-red'}">
                        {row['count_diff_pct']:+.1f}%
                    </td>
                    <td>{row['v1_win_rate']:.1f}%</td>
                    <td>{row['v2_win_rate']:.1f}%</td>
                    <td class="{'highlight-green' if row['win_rate_diff'] > 0 else 'highlight-red'}">
                        {row['win_rate_diff']:+.1f}%
                    </td>
                    <td>{row['v1_avg_return']:+.2f}%</td>
                    <td>{row['v2_avg_return']:+.2f}%</td>
                    <td class="{'highlight-green' if row['avg_return_diff'] > 0 else 'highlight-red'}">
                        {row['avg_return_diff']:+.2f}%
                    </td>
                </tr>
"""

        return f"""
    <div class="section">
        <h2>Sector Performance Analysis</h2>
        <p>Comparison showing how v2 sector weights (2025 India positioning - Defence, Capital Goods, Infrastructure focus) affect performance.</p>
        <table>
            <thead>
                <tr>
                    <th>Sector</th>
                    <th>v1 Count</th>
                    <th>v2 Count</th>
                    <th>Count Δ%</th>
                    <th>v1 Win Rate</th>
                    <th>v2 Win Rate</th>
                    <th>Win Rate Δ</th>
                    <th>v1 Avg Return</th>
                    <th>v2 Avg Return</th>
                    <th>Return Δ</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
"""

    def _build_risk_metrics_html(self, risk_metrics: Dict) -> str:
        """Build risk-adjusted returns section"""
        return f"""
    <div class="section">
        <h2>Risk-Adjusted Returns</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Return/Risk Ratio</h4>
                <div class="metric-value">{risk_metrics['v2_return_risk_ratio']:.3f}</div>
                <div class="metric-diff {'positive' if risk_metrics['return_risk_improvement_pct'] > 0 else 'negative'}">
                    {risk_metrics['return_risk_improvement_pct']:+.1f}% vs v1 ({risk_metrics['v1_return_risk_ratio']:.3f})
                </div>
            </div>
            <div class="metric-card">
                <h4>Sharpe Ratio</h4>
                <div class="metric-value">{risk_metrics['v2_sharpe_ratio']:.3f}</div>
                <div class="metric-diff {'positive' if risk_metrics['sharpe_improvement_pct'] > 0 else 'negative'}">
                    {risk_metrics['sharpe_improvement_pct']:+.1f}% vs v1 ({risk_metrics['v1_sharpe_ratio']:.3f})
                </div>
            </div>
            <div class="metric-card">
                <h4>Sortino Ratio</h4>
                <div class="metric-value">{risk_metrics['v2_sortino_ratio']:.3f}</div>
                <div class="metric-diff {'positive' if risk_metrics['sortino_improvement_pct'] > 0 else 'negative'}">
                    {risk_metrics['sortino_improvement_pct']:+.1f}% vs v1 ({risk_metrics['v1_sortino_ratio']:.3f})
                </div>
            </div>
            <div class="metric-card">
                <h4>Calmar Ratio</h4>
                <div class="metric-value">{risk_metrics['v2_calmar_ratio']:.3f}</div>
                <div class="metric-diff {'positive' if risk_metrics['calmar_improvement_pct'] > 0 else 'negative'}">
                    {risk_metrics['calmar_improvement_pct']:+.1f}% vs v1 ({risk_metrics['v1_calmar_ratio']:.3f})
                </div>
            </div>
        </div>
    </div>
"""

    def _build_filter_impact_html(self, filter_impact: Dict) -> str:
        """Build v2 filter impact section"""
        atr = filter_impact['atr_filter']
        liq = filter_impact['liquidity_filter']

        return f"""
    <div class="section">
        <h2>v2 Filter Impact Analysis</h2>

        <h3>Additional Candidates Captured</h3>
        <p>v2's relaxed filters and wider ranges captured <strong>{filter_impact['additional_candidates']:+d}</strong>
        additional trades (<strong>{filter_impact['additional_candidates_pct']:+.1f}%</strong> more than v1).</p>

        <h3>ATR Filter (Volatility >= 1.5%)</h3>
        <ul>
            <li>Filtered out: <strong>{atr.get('filtered_out', 0)}</strong> low-volatility candidates</li>
            <li>Average ATR%: <strong>{atr.get('avg_atr_pct', 0):.2f}%</strong></li>
            <li>Range: {atr.get('min_atr_pct', 0):.2f}% - {atr.get('max_atr_pct', 0):.2f}%</li>
        </ul>

        <h3>Liquidity Filter (Volume >= 500k shares)</h3>
        <ul>
            <li>Filtered out: <strong>{liq.get('filtered_out', 0)}</strong> low-liquidity candidates</li>
            <li>Average Volume: <strong>{liq.get('avg_volume', 0):,.0f}</strong> shares</li>
            <li>Range: {liq.get('min_volume', 0):,.0f} - {liq.get('max_volume', 0):,.0f} shares</li>
        </ul>

        <h3>Wider RSI Range Impact (30-70 vs 35-65)</h3>
        <p>Trades captured in wider range (RSI 30-35 or 65-70): <strong>{filter_impact['wider_rsi_range_trades']}</strong>
        ({filter_impact['wider_rsi_range_pct']:.1f}% of v2 trades)</p>
    </div>
"""

    def _build_distribution_html(self, distribution: Dict) -> str:
        """Build trade distribution section"""
        hold = distribution['hold_period']
        streaks = distribution['streaks']

        return f"""
    <div class="section">
        <h2>Trade Distribution Analysis</h2>

        <h3>Hold Period</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>v1</th>
                    <th>v2</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Average Hold Days</strong></td>
                    <td>{hold['v1_avg']:.1f}</td>
                    <td>{hold['v2_avg']:.1f}</td>
                    <td>{hold['v2_avg'] - hold['v1_avg']:+.1f}</td>
                </tr>
                <tr>
                    <td><strong>Median Hold Days</strong></td>
                    <td>{hold['v1_median']:.0f}</td>
                    <td>{hold['v2_median']:.0f}</td>
                    <td>{hold['v2_median'] - hold['v1_median']:+.0f}</td>
                </tr>
                <tr>
                    <td><strong>Std Deviation</strong></td>
                    <td>{hold['v1_std']:.1f}</td>
                    <td>{hold['v2_std']:.1f}</td>
                    <td>{hold['v2_std'] - hold['v1_std']:+.1f}</td>
                </tr>
            </tbody>
        </table>

        <h3>Win/Loss Streaks</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>v1</th>
                    <th>v2</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Max Win Streak</strong></td>
                    <td>{streaks['v1_max_win_streak']}</td>
                    <td>{streaks['v2_max_win_streak']}</td>
                    <td>{streaks['v2_max_win_streak'] - streaks['v1_max_win_streak']:+d}</td>
                </tr>
                <tr>
                    <td><strong>Max Loss Streak</strong></td>
                    <td>{streaks['v1_max_loss_streak']}</td>
                    <td>{streaks['v2_max_loss_streak']}</td>
                    <td>{streaks['v2_max_loss_streak'] - streaks['v1_max_loss_streak']:+d}</td>
                </tr>
            </tbody>
        </table>
    </div>
"""

    def _build_statistical_significance_html(self) -> str:
        """Build statistical significance section"""
        comp = self.comparison

        returns_sig_text = "YES - statistically significant" if comp.returns_statistically_significant else "NO - not statistically significant"
        win_rate_sig_text = "YES - statistically significant" if comp.win_rate_statistically_significant else "NO - not statistically significant"

        return f"""
    <div class="section">
        <h2>Statistical Significance</h2>

        <h3>Returns T-Test</h3>
        <p>Are v2 returns statistically different from v1?</p>
        <ul>
            <li>P-value: <strong>{comp.returns_ttest_pvalue:.4f}</strong></li>
            <li>Significance (p < 0.05): <strong>{returns_sig_text}</strong></li>
        </ul>

        <h3>Win Rate Chi-Square Test</h3>
        <p>Are v2 win rates statistically different from v1?</p>
        <ul>
            <li>P-value: <strong>{comp.win_rate_chi2_pvalue:.4f}</strong></li>
            <li>Significance (p < 0.05): <strong>{win_rate_sig_text}</strong></li>
        </ul>
    </div>
"""

    def _build_recommendation_html(self) -> str:
        """Build recommendation section"""
        comp = self.comparison

        confidence_badge = f'<span class="badge {comp.recommendation_confidence.lower()}">{comp.recommendation_confidence} CONFIDENCE</span>'

        reasons_html = ""
        for reason in comp.recommendation_reasons:
            reasons_html += f"<li>{reason}</li>"

        strategy_text = "Strategy v2 (Enhanced)" if comp.recommended_strategy == "v2" else \
                       "Strategy v1 (Current)" if comp.recommended_strategy == "v1" else \
                       "No Clear Winner - Further Analysis Needed"

        return f"""
    <div class="recommendation">
        <h2>Recommendation: {strategy_text} {confidence_badge}</h2>
        <p><strong>Analysis Reasoning:</strong></p>
        <ul>
            {reasons_html}
        </ul>
    </div>
"""

    def generate_txt_summary(self, output_path: str):
        """
        Generate executive summary in plain text format

        Args:
            output_path: Path to save TXT file
        """
        logger.info("Generating text summary...")

        comp = self.comparison

        summary = []
        summary.append("=" * 80)
        summary.append("STRATEGY COMPARISON EXECUTIVE SUMMARY")
        summary.append("v1 (Current) vs v2 (Enhanced) BTST Strategy")
        summary.append("=" * 80)
        summary.append("")
        summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Backtest Period: {self.v1_metrics['start_date']} to {self.v1_metrics['end_date']}")
        summary.append("")

        summary.append("KEY FINDINGS")
        summary.append("-" * 80)
        summary.append(f"Total Trades:        v1: {comp.total_trades_v1:>6}  |  v2: {comp.total_trades_v2:>6}  ({comp.trades_diff_pct:+7.1f}%)")
        summary.append(f"Win Rate:            v1: {comp.win_rate_v1:>5.1f}%  |  v2: {comp.win_rate_v2:>5.1f}%  ({comp.win_rate_diff_pct:+7.1f}%)")
        summary.append(f"Average Return:      v1: {comp.avg_return_v1:>+5.2f}%  |  v2: {comp.avg_return_v2:>+5.2f}%  ({comp.avg_return_diff_pct:+7.1f}%)")
        summary.append(f"Total Return:        v1: {comp.total_return_v1:>+6.1f}%  |  v2: {comp.total_return_v2:>+6.1f}%  ({comp.total_return_diff_pct:+7.1f}%)")
        summary.append(f"Profit Factor:       v1: {comp.profit_factor_v1:>6.2f}  |  v2: {comp.profit_factor_v2:>6.2f}  ({comp.profit_factor_diff_pct:+7.1f}%)")
        summary.append(f"Sharpe Ratio:        v1: {comp.sharpe_ratio_v1:>6.2f}  |  v2: {comp.sharpe_ratio_v2:>6.2f}  ({comp.sharpe_ratio_diff_pct:+7.1f}%)")
        summary.append(f"Max Drawdown:        v1: {comp.max_drawdown_v1:>6.2f}%  |  v2: {comp.max_drawdown_v2:>6.2f}%  ({comp.max_drawdown_diff_pct:+7.1f}%)")
        summary.append(f"Signals per Week:    v1: {comp.signals_per_week_v1:>6.2f}  |  v2: {comp.signals_per_week_v2:>6.2f}  ({comp.signals_per_week_diff_pct:+7.1f}%)")
        summary.append("")

        summary.append("STATISTICAL SIGNIFICANCE")
        summary.append("-" * 80)
        summary.append(f"Returns Difference:  {'SIGNIFICANT' if comp.returns_statistically_significant else 'NOT SIGNIFICANT'} (p={comp.returns_ttest_pvalue:.4f})")
        summary.append(f"Win Rate Difference: {'SIGNIFICANT' if comp.win_rate_statistically_significant else 'NOT SIGNIFICANT'} (p={comp.win_rate_chi2_pvalue:.4f})")
        summary.append("")

        summary.append("RECOMMENDATION")
        summary.append("-" * 80)
        strategy_text = "Strategy v2 (Enhanced)" if comp.recommended_strategy == "v2" else \
                       "Strategy v1 (Current)" if comp.recommended_strategy == "v1" else \
                       "No Clear Winner"
        summary.append(f"Recommended Strategy: {strategy_text}")
        summary.append(f"Confidence Level:     {comp.recommendation_confidence}")
        summary.append("")
        summary.append("Reasons:")
        for i, reason in enumerate(comp.recommendation_reasons, 1):
            summary.append(f"  {i}. {reason}")
        summary.append("")

        summary.append("=" * 80)
        summary.append("For detailed analysis, charts, and sector breakdowns, see HTML report.")
        summary.append("=" * 80)

        with open(output_path, 'w') as f:
            f.write("\n".join(summary))

        logger.info(f"Text summary saved: {output_path}")

    def generate_csv_report(self, output_path: str):
        """
        Generate detailed CSV with all metrics side-by-side

        Args:
            output_path: Path to save CSV file
        """
        logger.info("Generating CSV report...")

        comp = self.comparison

        # Build comprehensive metrics table
        data = [
            ['Metric', 'v1 Value', 'v2 Value', 'Difference', 'Percent Change'],
            ['Total Trades', comp.total_trades_v1, comp.total_trades_v2,
             comp.total_trades_v2 - comp.total_trades_v1, f"{comp.trades_diff_pct:.2f}%"],
            ['Win Rate (%)', f"{comp.win_rate_v1:.2f}", f"{comp.win_rate_v2:.2f}",
             f"{comp.win_rate_v2 - comp.win_rate_v1:.2f}", f"{comp.win_rate_diff_pct:.2f}%"],
            ['Average Return (%)', f"{comp.avg_return_v1:.2f}", f"{comp.avg_return_v2:.2f}",
             f"{comp.avg_return_v2 - comp.avg_return_v1:.2f}", f"{comp.avg_return_diff_pct:.2f}%"],
            ['Total Return (%)', f"{comp.total_return_v1:.2f}", f"{comp.total_return_v2:.2f}",
             f"{comp.total_return_v2 - comp.total_return_v1:.2f}", f"{comp.total_return_diff_pct:.2f}%"],
            ['Profit Factor', f"{comp.profit_factor_v1:.2f}", f"{comp.profit_factor_v2:.2f}",
             f"{comp.profit_factor_v2 - comp.profit_factor_v1:.2f}", f"{comp.profit_factor_diff_pct:.2f}%"],
            ['Sharpe Ratio', f"{comp.sharpe_ratio_v1:.2f}", f"{comp.sharpe_ratio_v2:.2f}",
             f"{comp.sharpe_ratio_v2 - comp.sharpe_ratio_v1:.2f}", f"{comp.sharpe_ratio_diff_pct:.2f}%"],
            ['Max Drawdown (%)', f"{comp.max_drawdown_v1:.2f}", f"{comp.max_drawdown_v2:.2f}",
             f"{comp.max_drawdown_v2 - comp.max_drawdown_v1:.2f}", f"{comp.max_drawdown_diff_pct:.2f}%"],
            ['Signals/Week', f"{comp.signals_per_week_v1:.2f}", f"{comp.signals_per_week_v2:.2f}",
             f"{comp.signals_per_week_v2 - comp.signals_per_week_v1:.2f}", f"{comp.signals_per_week_diff_pct:.2f}%"],
            ['', '', '', '', ''],
            ['Statistical Significance', '', '', '', ''],
            ['Returns T-Test P-Value', '', f"{comp.returns_ttest_pvalue:.4f}",
             'Significant' if comp.returns_statistically_significant else 'Not Significant', ''],
            ['Win Rate Chi2 P-Value', '', f"{comp.win_rate_chi2_pvalue:.4f}",
             'Significant' if comp.win_rate_statistically_significant else 'Not Significant', ''],
            ['', '', '', '', ''],
            ['Recommendation', '', '', '', ''],
            ['Recommended Strategy', '', comp.recommended_strategy, '', ''],
            ['Confidence Level', '', comp.recommendation_confidence, '', '']
        ]

        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_csv(output_path, index=False)

        logger.info(f"CSV report saved: {output_path}")

    def generate_all_reports(self, output_dir: str, formats: List[str] = ['html', 'txt', 'csv']):
        """
        Generate all comparison reports

        Args:
            output_dir: Directory to save reports
            formats: List of formats to generate ('html', 'txt', 'csv', or 'all')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        logger.info("=" * 80)
        logger.info("Generating comparison reports...")
        logger.info("=" * 80)

        generated_files = []

        if 'html' in formats or 'all' in formats:
            html_file = output_path / f'btst_comparison_report_{timestamp}.html'
            self.generate_html_report(str(html_file))
            generated_files.append(html_file)

        if 'txt' in formats or 'all' in formats:
            txt_file = output_path / f'btst_comparison_summary_{timestamp}.txt'
            self.generate_txt_summary(str(txt_file))
            generated_files.append(txt_file)

        if 'csv' in formats or 'all' in formats:
            csv_file = output_path / f'btst_comparison_metrics_{timestamp}.csv'
            self.generate_csv_report(str(csv_file))
            generated_files.append(csv_file)

        logger.info("=" * 80)
        logger.info("All reports generated successfully")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Generated files:")
        for f in generated_files:
            logger.info(f"  - {f.name}")
        logger.info("")

        return generated_files


# ============================================================================
# RESULTS TRACKING
# ============================================================================

def save_to_history_database(comparator: 'StrategyComparison', db_path: str, strategy_type: str = 'btst'):
    """
    Save backtest results and comparison to history database

    Args:
        comparator: StrategyComparison instance with loaded results
        db_path: Path to database file
        strategy_type: 'swing' or 'btst'
    """
    try:
        logger.info(f"Saving {strategy_type} comparison results to history database...")

        # Connect to database
        db = NiftyDatabaseManager(db_path)
        db.connect()

        # Create results tracking tables if they don't exist
        db.create_results_tracking_tables()

        run_date = datetime.now().strftime('%Y-%m-%d')

        # Save v1 results
        v1 = comparator.v1_metrics
        db.cursor.execute("""
            INSERT OR REPLACE INTO backtest_results_history (
                run_date, strategy_type, version,
                total_trades, winning_trades, losing_trades, win_rate,
                avg_return, total_return,
                sharpe_ratio, sortino_ratio, max_drawdown,
                avg_hold_days, median_hold_days, return_volatility,
                backtest_start_date, backtest_end_date, total_stocks_scanned,
                sector_performance, signal_distribution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date, strategy_type, 'v1',
            v1['total_trades'], v1['winning_trades'], v1['losing_trades'], v1['win_rate'],
            v1['avg_trade_pct'], v1['total_return_pct'],
            v1['sharpe_ratio'], v1.get('sortino_ratio', 0), v1['max_drawdown_pct'],
            v1.get('avg_hold_days', 1.0), v1.get('median_hold_days', 1.0), v1.get('return_volatility', 0),
            v1['start_date'], v1['end_date'], v1.get('total_stocks_scanned', 0),
            json.dumps(v1.get('sector_performance', {})),
            json.dumps(v1.get('entry_type_performance', {}))
        ))

        # Save v2 results
        v2 = comparator.v2_metrics
        db.cursor.execute("""
            INSERT OR REPLACE INTO backtest_results_history (
                run_date, strategy_type, version,
                total_trades, winning_trades, losing_trades, win_rate,
                avg_return, total_return,
                sharpe_ratio, sortino_ratio, max_drawdown,
                avg_hold_days, median_hold_days, return_volatility,
                backtest_start_date, backtest_end_date, total_stocks_scanned,
                sector_performance, signal_distribution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date, strategy_type, 'v2',
            v2['total_trades'], v2['winning_trades'], v2['losing_trades'], v2['win_rate'],
            v2['avg_trade_pct'], v2['total_return_pct'],
            v2['sharpe_ratio'], v2.get('sortino_ratio', 0), v2['max_drawdown_pct'],
            v2.get('avg_hold_days', 1.0), v2.get('median_hold_days', 1.0), v2.get('return_volatility', 0),
            v2['start_date'], v2['end_date'], v2.get('total_stocks_scanned', 0),
            json.dumps(v2.get('sector_performance', {})),
            json.dumps(v2.get('entry_type_performance', {}))
        ))

        # Save comparison results
        comp = comparator.comparison

        # Map recommendation to database format
        recommendation_map = {
            'v1': 'use_v1',
            'v2': 'use_v2',
            'NEUTRAL': 'inconclusive'
        }
        recommendation = recommendation_map.get(comp.recommended_strategy, 'inconclusive')

        # Count criteria met
        criteria_met = 0
        if comp.trades_diff_pct >= 30:
            criteria_met += 1
        if abs(comp.win_rate_v2 - comp.win_rate_v1) <= 5:
            criteria_met += 1
        if comp.avg_return_diff_pct >= 10:
            criteria_met += 1
        if comp.max_drawdown_v2 <= comp.max_drawdown_v1:
            criteria_met += 1
        if comp.sharpe_ratio_v2 > comp.sharpe_ratio_v1:
            criteria_met += 1

        db.cursor.execute("""
            INSERT OR REPLACE INTO comparison_results_history (
                run_date, strategy_type,
                recommendation, confidence_level, criteria_met, total_criteria,
                returns_ttest_pvalue, winrate_chisquare_pvalue,
                delta_total_trades, delta_win_rate, delta_avg_return,
                delta_sharpe_ratio, delta_max_drawdown,
                html_report_path, summary_report_path, csv_report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date, strategy_type,
            recommendation, comp.recommendation_confidence.lower(), criteria_met, 5,
            comp.returns_ttest_pvalue, comp.win_rate_chi2_pvalue,
            comp.total_trades_v2 - comp.total_trades_v1,
            comp.win_rate_v2 - comp.win_rate_v1,
            comp.avg_return_v2 - comp.avg_return_v1,
            comp.sharpe_ratio_v2 - comp.sharpe_ratio_v1,
            comp.max_drawdown_v2 - comp.max_drawdown_v1,
            '', '', ''  # Report paths will be updated later if needed
        ))

        db.conn.commit()
        db.close()

        logger.info(f"Successfully saved {strategy_type} results to history database")

    except Exception as e:
        logger.error(f"Error saving to history database: {e}", exc_info=True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Compare v1 vs v2 Backtesting Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all reports (HTML, TXT, CSV)
  python compare_strategies.py \\
      --v1-metrics results/backtest_v1_metrics_20251111_120000.json \\
      --v1-trades results/backtest_v1_trades_20251111_120000.csv \\
      --v2-metrics results/backtest_v2_metrics_20251111_130000.json \\
      --v2-trades results/backtest_v2_trades_20251111_130000.csv \\
      --output-dir ./comparison_reports

  # Generate only HTML report
  python compare_strategies.py \\
      --v1-metrics results/v1_metrics.json \\
      --v1-trades results/v1_trades.csv \\
      --v2-metrics results/v2_metrics.json \\
      --v2-trades results/v2_trades.csv \\
      --format html
        """
    )

    parser.add_argument(
        '--v1-metrics',
        type=str,
        required=True,
        help='Path to v1 metrics JSON file'
    )

    parser.add_argument(
        '--v1-trades',
        type=str,
        required=True,
        help='Path to v1 trades CSV file'
    )

    parser.add_argument(
        '--v2-metrics',
        type=str,
        required=True,
        help='Path to v2 metrics JSON file'
    )

    parser.add_argument(
        '--v2-trades',
        type=str,
        required=True,
        help='Path to v2 trades CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/backtesting/comparison_reports',
        help='Output directory for comparison reports (default: ./comparison_reports)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'txt', 'csv', 'all'],
        default='all',
        help='Output format (default: all)'
    )

    args = parser.parse_args()

    try:
        # Initialize comparison engine
        comparator = StrategyComparison(
            v1_metrics_path=args.v1_metrics,
            v1_trades_path=args.v1_trades,
            v2_metrics_path=args.v2_metrics,
            v2_trades_path=args.v2_trades
        )

        # Load results
        comparator.load_backtest_results()

        # Perform comparison
        comparator.compare_metrics()

        # Generate reports
        formats = [args.format] if args.format != 'all' else ['html', 'txt', 'csv']
        generated_files = comparator.generate_all_reports(
            output_dir=args.output_dir,
            formats=formats
        )

        # Save to history database
        db_path = str(Path(__file__).parent.parent / "data" / "nifty500_historical.db")
        save_to_history_database(comparator, db_path, strategy_type='btst')

        # Print recommendation to console
        comp = comparator.comparison
        print("\n")
        print("=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        strategy_text = "Strategy v2 (Enhanced)" if comp.recommended_strategy == "v2" else \
                       "Strategy v1 (Current)" if comp.recommended_strategy == "v1" else \
                       "No Clear Winner"
        print(f"Recommended Strategy: {strategy_text}")
        print(f"Confidence Level:     {comp.recommendation_confidence}")
        print("")
        print("Key Reasons:")
        for i, reason in enumerate(comp.recommendation_reasons, 1):
            print(f"  {i}. {reason}")
        print("=" * 80)
        print("")
        print(f"Reports saved to: {args.output_dir}")
        print("")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
