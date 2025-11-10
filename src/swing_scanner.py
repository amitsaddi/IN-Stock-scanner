"""
Swing Trading Scanner - Identifies swing trading opportunities
"""
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

from config import Config, SWING_PREFERRED_SECTORS
from data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SwingCandidate:
    """Data class for swing trading candidate"""
    symbol: str
    current_price: float
    market_cap: float
    sector: str
    rsi: float
    macd_signal: str
    ema_alignment: bool
    volume_ratio: float
    debt_to_equity: float
    roe: float
    score: float
    entry_type: str  # pullback, breakout, macd_cross, etc.
    target: float
    stop_loss: float
    reason: str


class SwingScanner:
    """Scanner for swing trading opportunities"""
    
    def __init__(self):
        self.config = Config()
        self.fetcher = DataFetcher()
        self.criteria = self.config.SWING
    
    def check_swing_criteria(self, historical_df: pd.DataFrame, 
                            fundamentals: Dict) -> tuple[bool, float, str, str]:
        """
        Check if stock meets swing trading criteria
        
        Returns:
            (passes, score, entry_type, reason)
        """
        if historical_df is None or historical_df.empty or len(historical_df) < 50:
            return False, 0, "", "Insufficient data"
        
        reasons = []
        score = 0
        entry_type = "none"
        
        latest = historical_df.iloc[-1]
        prev = historical_df.iloc[-2]
        
        # 1. Fundamental filters (must pass all)
        market_cap = fundamentals.get('market_cap', 0)
        debt_to_equity = fundamentals.get('debt_to_equity', 999)
        roe = fundamentals.get('roe', 0)
        sector = fundamentals.get('sector', 'Unknown')
        
        if market_cap < self.criteria.min_market_cap:
            return False, 0, "", f"Market cap too small: ₹{market_cap:.0f} Cr"
        
        if debt_to_equity > self.criteria.max_debt_to_equity:
            return False, 0, "", f"High debt: {debt_to_equity:.2f}"
        
        if roe < self.criteria.min_roe:
            return False, 0, "", f"Low ROE: {roe:.1f}%"
        
        reasons.append(f"✓ Fundamentals: MCap ₹{market_cap:.0f}Cr, D/E {debt_to_equity:.2f}, ROE {roe:.1f}%")
        score += 20
        
        # 2. Trend alignment (above all MAs) - 20 points
        ema_20 = latest.get('EMA_20', 0)
        ema_50 = latest.get('EMA_50', 0)
        sma_200 = latest.get('SMA_200', 0)
        close = latest['Close']
        
        above_all_ma = close > ema_20 > ema_50 > sma_200
        
        if above_all_ma:
            score += 20
            reasons.append("✓ Above all MAs (strong uptrend)")
        elif close > ema_20 and close > ema_50:
            score += 15
            reasons.append("⚠ Above 20/50 EMA (moderate trend)")
        else:
            score += 5
            reasons.append("⚠ Not in clear uptrend")
        
        # 3. RSI check (40-60 ideal) - 20 points
        rsi = latest.get('RSI', 50)
        
        if self.criteria.rsi_min <= rsi <= self.criteria.rsi_max:
            score += 20
            reasons.append(f"✓ RSI {rsi:.1f} (ideal range)")
        elif 30 < rsi < 40:
            # Pullback opportunity
            score += 15
            entry_type = "pullback"
            reasons.append(f"✓ RSI {rsi:.1f} (pullback zone)")
        elif rsi > 60:
            score += 5
            reasons.append(f"⚠ RSI {rsi:.1f} (approaching overbought)")
        else:
            reasons.append(f"⚠ RSI {rsi:.1f}")
        
        # 4. MACD signals - 15 points
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_hist = latest.get('MACD_Hist', 0)
        prev_macd_hist = prev.get('MACD_Hist', 0)
        
        macd_status = "neutral"
        if macd > macd_signal and macd_hist > 0:
            score += 15
            macd_status = "bullish"
            reasons.append("✓ MACD bullish")
            if prev_macd_hist <= 0:  # Fresh crossover
                entry_type = "macd_cross"
        elif macd > macd_signal:
            score += 10
            macd_status = "neutral"
            reasons.append("⚠ MACD neutral")
        
        # 5. Volume confirmation - 15 points
        volume_ratio = latest.get('Volume_Ratio', 1.0)
        
        if volume_ratio >= self.criteria.min_volume_multiplier:
            score += 15
            reasons.append(f"✓ Volume {volume_ratio:.1f}x average")
        else:
            score += 5
            reasons.append(f"⚠ Volume {volume_ratio:.1f}x")
        
        # 6. Sector preference - 10 points
        if sector in SWING_PREFERRED_SECTORS:
            score += 10
            reasons.append(f"✓ Preferred sector: {sector}")
        else:
            score += 5
            reasons.append(f"Sector: {sector}")
        
        # Detect entry type if not already set
        if entry_type == "none":
            # Check for breakout
            if close > latest.get('EMA_20', 0) * 1.02 and volume_ratio > 1.5:
                entry_type = "breakout"
            # Check for MA crossover
            elif ema_20 > ema_50 and prev.get('EMA_20', 0) <= prev.get('EMA_50', 0):
                entry_type = "ma_cross"
            else:
                entry_type = "trend_follow"
        
        # Minimum score threshold: 65/100
        passes = score >= 65
        
        return passes, score, entry_type, "; ".join(reasons)
    
    def calculate_targets(self, current_price: float, entry_type: str) -> tuple[float, float]:
        """
        Calculate target and stop loss based on entry type
        
        Returns:
            (target, stop_loss)
        """
        if entry_type == "breakout":
            target = current_price * 1.12  # 12% target
            stop_loss = current_price * 0.95  # 5% SL
        elif entry_type == "pullback":
            target = current_price * 1.10  # 10% target
            stop_loss = current_price * 0.93  # 7% SL
        else:  # trend_follow, macd_cross, ma_cross
            target = current_price * 1.12  # 12% target
            stop_loss = current_price * 0.94  # 6% SL
        
        return round(target, 2), round(stop_loss, 2)
    
    def scan_for_swing(self, symbols: List[str] = None, max_results: int = None) -> List[SwingCandidate]:
        """
        Scan stocks for swing trading opportunities
        
        Args:
            symbols: List of symbols to scan
            max_results: Maximum results to return
        
        Returns:
            List of SwingCandidate objects sorted by score
        """
        if symbols is None:
            symbols = self.fetcher.get_nifty500_symbols()
        
        if max_results is None:
            max_results = self.criteria.max_results
        
        logger.info(f"Scanning {len(symbols)} stocks for swing opportunities...")
        
        candidates = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})")
                
                # Fetch fundamentals first (quick filter)
                fundamentals = self.fetcher.fetch_fundamentals(symbol)
                if fundamentals is None:
                    continue
                
                # Quick filter on fundamentals
                if fundamentals.get('market_cap', 0) < self.criteria.min_market_cap:
                    continue
                if fundamentals.get('debt_to_equity', 999) > self.criteria.max_debt_to_equity:
                    continue
                
                # Fetch historical data with indicators
                historical_df = self.fetcher.fetch_daily_data(symbol, days=100)
                if historical_df is None:
                    continue
                
                historical_df = self.fetcher.calculate_technical_indicators(historical_df)
                
                # Check criteria
                passes, score, entry_type, reason = self.check_swing_criteria(
                    historical_df, fundamentals
                )
                
                if passes:
                    latest = historical_df.iloc[-1]
                    current_price = latest['Close']
                    
                    target, stop_loss = self.calculate_targets(current_price, entry_type)
                    
                    candidate = SwingCandidate(
                        symbol=symbol.replace('.NS', ''),
                        current_price=round(current_price, 2),
                        market_cap=fundamentals['market_cap'],
                        sector=fundamentals['sector'],
                        rsi=round(latest.get('RSI', 0), 1),
                        macd_signal="bullish" if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else "bearish",
                        ema_alignment=current_price > latest.get('EMA_20', 0) > latest.get('EMA_50', 0),
                        volume_ratio=round(latest.get('Volume_Ratio', 1.0), 1),
                        debt_to_equity=fundamentals['debt_to_equity'],
                        roe=fundamentals['roe'],
                        score=score,
                        entry_type=entry_type,
                        target=target,
                        stop_loss=stop_loss,
                        reason=reason
                    )
                    candidates.append(candidate)
                    logger.info(f"✓ Swing candidate: {candidate.symbol} (score: {score:.0f})")
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:max_results]
    
    def generate_report(self, candidates: List[SwingCandidate]) -> str:
        """Generate text report of swing candidates"""
        if not candidates:
            return "No swing trading candidates found."
        
        report = f"🟠 SWING TRADING WATCHLIST - {datetime.now().strftime('%d %B %Y')}\n"
        report += "=" * 70 + "\n\n"
        
        for i, c in enumerate(candidates, 1):
            report += f"{i}. {c.symbol} - {c.entry_type.upper()}\n"
            report += f"   Price: ₹{c.current_price} | Sector: {c.sector}\n"
            report += f"   RSI: {c.rsi} | MACD: {c.macd_signal} | Volume: {c.volume_ratio}x\n"
            report += f"   ROE: {c.roe:.1f}% | D/E: {c.debt_to_equity:.2f} | MCap: ₹{c.market_cap:.0f}Cr\n"
            report += f"   Target: ₹{c.target} ({((c.target/c.current_price-1)*100):.1f}%) | SL: ₹{c.stop_loss}\n"
            report += f"   Score: {c.score:.0f}/100\n"
            report += f"   {c.reason}\n"
            report += "\n"
        
        report += "📌 SWING TRADING TIPS:\n"
        report += "• Hold for 3-15 days\n"
        report += "• Book 50% at first target, trail rest\n"
        report += "• Exit ruthlessly if SL hit\n"
        report += "• Review positions daily after market close\n"
        
        return report
    
    def save_results(self, candidates: List[SwingCandidate], filename: str = None):
        """Save scan results to CSV"""
        if not candidates:
            logger.info("No candidates to save")
            return
        
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"swing_scan_{date_str}.csv"
        
        import os
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(self.config.RESULTS_DIR, filename)
        
        data = []
        for c in candidates:
            data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Symbol': c.symbol,
                'Price': c.current_price,
                'Sector': c.sector,
                'Entry_Type': c.entry_type,
                'RSI': c.rsi,
                'Volume_Ratio': c.volume_ratio,
                'ROE_%': c.roe,
                'Debt_Equity': c.debt_to_equity,
                'Target': c.target,
                'Stop_Loss': c.stop_loss,
                'Score': c.score
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")


def run_swing_scan(test_mode: bool = False):
    """Main function to run swing scan"""
    scanner = SwingScanner()
    
    if test_mode:
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS", "LT.NS",
                   "BHARTIARTL.NS", "TATAMOTORS.NS", "INFY.NS", "KOTAKBANK.NS", "ITC.NS"]
        logger.info("Running in TEST MODE with 10 stocks")
    else:
        symbols = None
    
    candidates = scanner.scan_for_swing(symbols=symbols)
    report = scanner.generate_report(candidates)
    print("\n" + report)
    scanner.save_results(candidates)
    
    return candidates, report


if __name__ == "__main__":
    run_swing_scan(test_mode=True)
