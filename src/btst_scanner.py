"""
BTST Scanner - Identifies Buy Today Sell Tomorrow opportunities
"""
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

from config import Config, BTST_EXCLUDED_SECTORS
from data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BTSTCandidate:
    """Data class for BTST candidate"""
    symbol: str
    current_price: float
    day_change_pct: float
    volume_ratio: float
    high_proximity_pct: float
    sector: str
    score: float
    reason: str


class BTSTScanner:
    """Scanner for BTST opportunities"""
    
    def __init__(self):
        self.config = Config()
        self.fetcher = DataFetcher()
        self.criteria = self.config.BTST
    
    def check_btst_criteria(self, current_data: Dict, historical_df: pd.DataFrame, 
                           fundamentals: Dict) -> tuple[bool, float, str]:
        """
        Check if stock meets BTST criteria
        
        Returns:
            (passes, score, reason)
        """
        reasons = []
        score = 0
        max_score = 100
        
        # 1. Price gain (2-3% at 3 PM) - 30 points
        day_change = current_data['day_change_pct']
        if self.criteria.min_gain_percent <= day_change <= self.criteria.max_gain_percent:
            gain_score = 30
            score += gain_score
            reasons.append(f"✓ Gain {day_change:.1f}% (ideal 2-3%)")
        elif day_change > self.criteria.max_gain_percent:
            # Too much gain might mean overbought
            gain_score = 15
            score += gain_score
            reasons.append(f"⚠ Gain {day_change:.1f}% (>3%, may be overbought)")
        else:
            reasons.append(f"✗ Gain {day_change:.1f}% (below 2%)")
            return False, 0, "; ".join(reasons)
        
        # 2. Volume (1.5x average) - 25 points
        if historical_df is not None and not historical_df.empty:
            latest = historical_df.iloc[-1]
            if 'Volume_Ratio' in historical_df.columns:
                volume_ratio = latest['Volume_Ratio']
                if volume_ratio >= self.criteria.min_volume_multiplier:
                    vol_score = 25
                    score += vol_score
                    reasons.append(f"✓ Volume {volume_ratio:.1f}x average")
                else:
                    vol_score = 10
                    score += vol_score
                    reasons.append(f"⚠ Volume {volume_ratio:.1f}x (below 1.5x)")
        
        # 3. Closing near day high - 20 points
        high_prox = current_data['high_proximity_pct']
        if high_prox >= 90:
            score += 20
            reasons.append(f"✓ Closing near high ({high_prox:.0f}%)")
        elif high_prox >= 80:
            score += 15
            reasons.append(f"⚠ Near high ({high_prox:.0f}%)")
        else:
            score += 5
            reasons.append(f"⚠ Not near high ({high_prox:.0f}%)")
        
        # 4. Trend confirmation (above 20 EMA) - 15 points
        if historical_df is not None and 'EMA_20' in historical_df.columns:
            latest = historical_df.iloc[-1]
            if latest['Close'] > latest['EMA_20']:
                score += 15
                reasons.append("✓ Above 20 EMA (uptrend)")
            else:
                score += 5
                reasons.append("⚠ Below 20 EMA")
        
        # 5. Sector check - 10 points
        sector = fundamentals.get('sector', 'Unknown')
        if sector not in BTST_EXCLUDED_SECTORS:
            score += 10
            reasons.append(f"✓ Sector: {sector}")
        else:
            reasons.append(f"⚠ Excluded sector: {sector}")
        
        # Minimum score threshold: 60/100
        passes = score >= 60
        
        return passes, score, "; ".join(reasons)
    
    def scan_for_btst(self, symbols: List[str] = None, max_results: int = None) -> List[BTSTCandidate]:
        """
        Scan stocks for BTST opportunities
        
        Args:
            symbols: List of symbols to scan (if None, scans Nifty 500)
            max_results: Maximum number of results to return
        
        Returns:
            List of BTSTCandidate objects sorted by score
        """
        if symbols is None:
            symbols = self.fetcher.get_nifty500_symbols()
        
        if max_results is None:
            max_results = self.criteria.max_results
        
        logger.info(f"Scanning {len(symbols)} stocks for BTST opportunities...")
        
        candidates = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})")
                
                # Fetch current data
                current_data = self.fetcher.get_current_price_and_volume(symbol)
                if current_data is None:
                    continue
                
                # Quick filter: Must have 2%+ gain
                if current_data['day_change_pct'] < self.criteria.min_gain_percent:
                    continue
                
                # Fetch historical data
                historical_df = self.fetcher.fetch_daily_data(symbol, days=50)
                if historical_df is None:
                    continue
                
                historical_df = self.fetcher.calculate_technical_indicators(historical_df)
                
                # Fetch fundamentals
                fundamentals = self.fetcher.fetch_fundamentals(symbol)
                if fundamentals is None:
                    fundamentals = {'sector': 'Unknown'}
                
                # Check criteria
                passes, score, reason = self.check_btst_criteria(
                    current_data, historical_df, fundamentals
                )
                
                if passes:
                    # Calculate volume ratio
                    volume_ratio = historical_df.iloc[-1].get('Volume_Ratio', 1.0) if not historical_df.empty else 1.0
                    
                    candidate = BTSTCandidate(
                        symbol=current_data['symbol'],
                        current_price=current_data['current_price'],
                        day_change_pct=current_data['day_change_pct'],
                        volume_ratio=volume_ratio,
                        high_proximity_pct=current_data['high_proximity_pct'],
                        sector=fundamentals['sector'],
                        score=score,
                        reason=reason
                    )
                    candidates.append(candidate)
                    logger.info(f"✓ BTST candidate found: {current_data['symbol']} (score: {score:.0f})")
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Return top N
        return candidates[:max_results]
    
    def generate_report(self, candidates: List[BTSTCandidate]) -> str:
        """Generate a text report of BTST candidates"""
        if not candidates:
            return "No BTST candidates found today."
        
        report = f"🟢 BTST OPPORTUNITIES - {datetime.now().strftime('%d %B %Y, %I:%M %p')}\n"
        report += "=" * 70 + "\n\n"
        
        for i, candidate in enumerate(candidates, 1):
            report += f"{i}. {candidate.symbol}\n"
            report += f"   Price: ₹{candidate.current_price} | Gain: {candidate.day_change_pct:+.2f}%\n"
            report += f"   Volume: {candidate.volume_ratio:.1f}x | Near High: {candidate.high_proximity_pct:.0f}%\n"
            report += f"   Sector: {candidate.sector} | Score: {candidate.score:.0f}/100\n"
            report += f"   Analysis: {candidate.reason}\n"
            report += f"   Entry: Buy at 3:00-3:20 PM | Target: {candidate.current_price * 1.02:.2f} (2%)\n"
            report += f"   Stop Loss: {candidate.current_price * 0.98:.2f} (2%)\n"
            report += "\n"
        
        report += "⚠️ IMPORTANT REMINDERS:\n"
        report += "• Enter positions between 3:00-3:20 PM\n"
        report += "• Exit in first 15 minutes next morning\n"
        report += "• Never hold if gap down - exit immediately\n"
        report += "• Max 2-3 BTST positions simultaneously\n"
        
        return report
    
    def save_results(self, candidates: List[BTSTCandidate], filename: str = None):
        """Save scan results to CSV"""
        if not candidates:
            logger.info("No candidates to save")
            return
        
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"btst_scan_{date_str}.csv"
        
        import os
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(self.config.RESULTS_DIR, filename)
        
        # Convert to DataFrame
        data = []
        for c in candidates:
            data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Time': datetime.now().strftime('%H:%M'),
                'Symbol': c.symbol,
                'Price': c.current_price,
                'Day_Change_%': c.day_change_pct,
                'Volume_Ratio': c.volume_ratio,
                'High_Proximity_%': c.high_proximity_pct,
                'Sector': c.sector,
                'Score': c.score,
                'Target': round(c.current_price * 1.02, 2),
                'Stop_Loss': round(c.current_price * 0.98, 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")


def run_btst_scan(test_mode: bool = False):
    """
    Main function to run BTST scan
    
    Args:
        test_mode: If True, only scan a small subset of stocks
    """
    scanner = BTSTScanner()
    
    # Get symbols
    if test_mode:
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", 
                   "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS"]
        logger.info("Running in TEST MODE with 10 stocks")
    else:
        symbols = None  # Will fetch all Nifty 500
    
    # Run scan
    candidates = scanner.scan_for_btst(symbols=symbols)
    
    # Generate report
    report = scanner.generate_report(candidates)
    print("\n" + report)
    
    # Save results
    scanner.save_results(candidates)
    
    return candidates, report


if __name__ == "__main__":
    run_btst_scan(test_mode=True)
