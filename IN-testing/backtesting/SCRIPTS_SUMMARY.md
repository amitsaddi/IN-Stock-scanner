# IN-testing Backtesting Scripts Summary

## Overview
This directory contains 6 backtesting scripts for Indian market (Nifty 500) strategies.

## Backtest Engines (4 scripts)

### 1. swing_backtest_v1.py (1,253 lines) ✅ ALREADY COMPLETE
**Strategy:** Current production swing trading (strict filters)
- **Fundamentals:** D/E ≤ 0.5, ROE ≥ 15%, Market Cap ≥ ₹5000 Cr
- **Technical:** RSI 40-60, Volume ≥ 1.2x
- **Entry Signals:** 5 signals (Pullback, Breakout, MACD Cross, MA Cross, Trend Follow)
- **Hold Period:** 3-15 days
- **Min Score:** 60
- **Max Candidates:** 15/scan

**Usage:**
```bash
python swing_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01
```

---

### 2. swing_backtest_v2.py (1,345 lines) ✅ NEW
**Strategy:** Enhanced swing trading (relaxed filters + 3 new signals)
- **Fundamentals:** D/E ≤ 0.8 (↑), ROE ≥ 12% (↓)
- **Technical:** RSI 30-70 (wider), Volume ≥ 1.0x (↓), ATR ≥ 1.5% (NEW), Min Vol 500k (NEW)
- **Entry Signals:** 8 signals (5 original + Bollinger Bounce, Volume Surge, Consolidation Breakout)
- **Hold Period:** 3-15 days
- **Min Score:** 55 (↓)
- **Max Candidates:** 20/scan (↑)

**Key Changes from v1:**
- Relaxed D/E: 0.5 → 0.8
- Relaxed ROE: 15% → 12%
- Wider RSI: 40-60 → 30-70
- Relaxed volume: 1.2x → 1.0x
- NEW ATR filter (1.5% min volatility)
- NEW Liquidity filter (500k shares min)
- 3 NEW entry signals
- Rebalanced scoring (more momentum/technical focus)

**Usage:**
```bash
python swing_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01
```

---

### 3. btst_backtest_v1.py (852 lines) ✅ NEW
**Strategy:** BTST v1 (Buy Today Sell Tomorrow - strict)
- **Entry:** 3:15 PM (market close)
- **Exit:** 9:15 AM next day (market open)
- **Hold:** Always 1 day
- **Criteria:**
  - Day gain: 2.0-3.5%
  - Volume: ≥ 1.5x
  - High proximity: ≥ 90%
  - Above 20 EMA
  - Positive MACD & RSI > 50
- **Min Score:** 60
- **Max Candidates:** 10/day

**Usage:**
```bash
python btst_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01
```

---

### 4. btst_backtest_v2.py (871 lines) ✅ NEW
**Strategy:** BTST v2 (relaxed filters + new indicators)
- **Entry:** 3:15 PM (market close)
- **Exit:** 9:15 AM next day (market open)
- **Hold:** Always 1 day
- **Criteria (RELAXED):**
  - Day gain: 1.5-4.0% (↑)
  - Volume: ≥ 1.3x (↓)
  - High proximity: ≥ 85% (↓)
  - RSI < 70 (NEW)
  - MACD histogram > 0 (NEW)
  - ATR ≥ 1.0% (NEW)
- **Min Score:** 55 (↓)
- **Max Candidates:** 15/day (↑)

**Key Changes from v1:**
- Relaxed day gain: 2.0-3.5% → 1.5-4.0%
- Relaxed volume: 1.5x → 1.3x
- Relaxed high proximity: 90% → 85%
- NEW RSI filter (< 70)
- NEW MACD histogram filter
- NEW ATR filter (≥ 1.0%)
- Lowered min score: 60 → 55
- More candidates: 10 → 15

**Usage:**
```bash
python btst_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01
```

---

## Comparison Scripts (2 scripts)

### 5. compare_swing_strategies.py (1,585 lines) ✅ NEW
**Purpose:** Compare swing_backtest_v1 vs swing_backtest_v2 results

**Analysis Includes:**
- Overall performance comparison (win rate, returns, Sharpe ratio)
- Entry signal analysis (including v2's 3 new signals)
- Sector performance breakdown
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Statistical significance tests (T-test for returns, Chi-square for win rates)
- Hold period distribution
- Return distribution
- Automated recommendations

**Outputs:**
- HTML report (interactive, charts)
- TXT summary (executive summary)
- CSV metrics (detailed comparison table)

**Usage:**
```bash
python compare_swing_strategies.py \
  --v1-metrics results/swing_backtest_v1_metrics_*.json \
  --v1-trades results/swing_backtest_v1_trades_*.csv \
  --v2-metrics results/swing_backtest_v2_metrics_*.json \
  --v2-trades results/swing_backtest_v2_trades_*.csv \
  --output-dir comparison_reports
```

---

### 6. compare_btst_strategies.py (1,585 lines) ✅ NEW
**Purpose:** Compare btst_backtest_v1 vs btst_backtest_v2 results

**Analysis Includes:**
- All standard comparison features (same as swing)
- **PLUS BTST-specific gap analysis:**
  - Gap-up rate and average
  - Gap-down rate and average
  - Gap distribution charts
  - Overnight behavior patterns

**Outputs:**
- HTML report (with gap analysis section)
- TXT summary
- CSV metrics

**Usage:**
```bash
python compare_btst_strategies.py \
  --v1-metrics results/btst_backtest_v1_metrics_*.json \
  --v1-trades results/btst_backtest_v1_trades_*.csv \
  --v2-metrics results/btst_backtest_v2_metrics_*.json \
  --v2-trades results/btst_backtest_v2_trades_*.csv \
  --output-dir comparison_reports
```

---

## File Structure

```
IN-testing/backtesting/
├── swing_backtest_v1.py       (1,253 lines) - Swing v1 engine ✅
├── swing_backtest_v2.py       (1,345 lines) - Swing v2 engine ✅ NEW
├── btst_backtest_v1.py        (852 lines)   - BTST v1 engine ✅ NEW
├── btst_backtest_v2.py        (871 lines)   - BTST v2 engine ✅ NEW
├── compare_swing_strategies.py (1,585 lines) - Swing comparison ✅ NEW
├── compare_btst_strategies.py  (1,585 lines) - BTST comparison ✅ NEW
└── results/                    - Output directory
    ├── swing_backtest_v1_*.csv/json/txt
    ├── swing_backtest_v2_*.csv/json/txt
    ├── btst_backtest_v1_*.csv/json/txt
    ├── btst_backtest_v2_*.csv/json/txt
    └── comparison_reports/
```

**Total:** 7,491 lines of code across 6 scripts (5 new + 1 existing)

---

## Quick Start Workflow

### 1. Run all backtests
```bash
# Swing strategies
python swing_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01
python swing_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01

# BTST strategies
python btst_backtest_v1.py --start-date 2025-05-01 --end-date 2025-11-01
python btst_backtest_v2.py --start-date 2025-05-01 --end-date 2025-11-01
```

### 2. Compare results
```bash
# Compare swing strategies
python compare_swing_strategies.py \
  --v1-metrics results/swing_backtest_v1_metrics_*.json \
  --v1-trades results/swing_backtest_v1_trades_*.csv \
  --v2-metrics results/swing_backtest_v2_metrics_*.json \
  --v2-trades results/swing_backtest_v2_trades_*.csv

# Compare BTST strategies
python compare_btst_strategies.py \
  --v1-metrics results/btst_backtest_v1_metrics_*.json \
  --v1-trades results/btst_backtest_v1_trades_*.csv \
  --v2-metrics results/btst_backtest_v2_metrics_*.json \
  --v2-trades results/btst_backtest_v2_trades_*.csv
```

### 3. Review results
- Open HTML reports in browser
- Check TXT summaries for quick insights
- Analyze CSV files for detailed data

---

## Key Features

### All Backtest Engines Include:
- ✅ SQLite database integration
- ✅ Complete trade tracking (entry/exit/P&L)
- ✅ Risk management (targets/stop-loss)
- ✅ Performance metrics (win rate, Sharpe, drawdown)
- ✅ Sector analysis
- ✅ Entry signal breakdown
- ✅ Return distribution
- ✅ CSV/JSON/TXT output formats
- ✅ Verbose logging option
- ✅ Progress tracking

### All Comparison Scripts Include:
- ✅ Side-by-side performance comparison
- ✅ Statistical significance testing
- ✅ Risk-adjusted returns analysis
- ✅ Interactive HTML reports
- ✅ Executive TXT summaries
- ✅ Detailed CSV exports
- ✅ Automated recommendations

---

## Database Requirements

All scripts require:
- **Database:** `/Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/data/nifty500_historical.db`
- **Tables:** `stock_metadata` + individual stock tables (`stock_SYMBOL`)
- **Columns Required:**
  - OHLCV: date, open, high, low, close, volume
  - Indicators: ema_20, ema_50, sma_200, rsi, macd, macd_signal, macd_hist
  - Metrics: volume_ratio, week_52_high, week_52_high_proximity
  - Bollinger: bb_upper, bb_middle, bb_lower (for v2)
  - ATR: atr (for v2)

---

## Notes

1. **swing_backtest_v1.py** was already complete (1,253 lines)
2. **5 NEW scripts** created totaling 6,238 lines
3. All scripts follow same patterns/structure for consistency
4. All scripts have valid Python syntax (verified)
5. Reused classes, methods, and patterns from swing_backtest_v1.py
6. BTST scripts are shorter (~850 lines) due to simpler 1-day strategy
7. Comparison scripts are identical in structure, with BTST version having additional gap analysis

---

Generated: 2025-11-11
