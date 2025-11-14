# IN-testing: Indian Stock Market Backtesting Framework

## Quick Start - One Command!

```bash
cd /Users/amitsaddi/Documents/git/IN-Stock-scanner/IN-testing/backtesting
./run_full_backtest.sh --test
```

That's it! This single command will:
1. Download historical data for 10 representative Indian stocks (5 minutes)
2. Run Swing v1 backtest (current strategy)
3. Run Swing v2 backtest (enhanced strategy)
4. Run BTST v1 backtest (current strategy)
5. Run BTST v2 backtest (enhanced strategy)
6. Generate comprehensive comparison reports
7. Open results in your browser

## What to Expect

### Test Mode (--test flag)
- **Runtime**: ~5-10 minutes total
- **Stocks**: 10 diverse Nifty 500 stocks
- **Period**: Last 180 days
- **Output**: HTML reports, CSV data, text summaries

### Full Mode (no --test flag)
- **Runtime**: ~15-25 minutes total
- **Stocks**: Top 200 Nifty 500 stocks by market cap
- **Period**: Last 180 days  
- **Output**: Comprehensive analysis across all stocks

## Understanding the Results

### Swing Trading Results
- **Hold Period**: 3-15 days
- **Target Returns**: 10-15%
- **v1 Strategy**: Conservative (60+ score, tighter filters)
- **v2 Strategy**: Enhanced (55+ score, relaxed filters, 3 new signals)

**Key Metrics to Watch**:
- Total signals per week (more is better for v2)
- Win rate (should be 50%+)
- Average return per trade
- Sharpe ratio (risk-adjusted returns)

### BTST Results
- **Hold Period**: Overnight (buy 3:15 PM, sell 9:15 AM next day)
- **Target Returns**: Gap up on next day open
- **v1 Strategy**: Strict criteria (60+ score, 2-3% day gain)
- **v2 Strategy**: Enhanced (55+ score, 1.5-4% gain range)

**Key Metrics to Watch**:
- Gap-up success rate
- Average overnight return
- Risk/reward ratio
- Signal frequency

## Interpretation Guide

### When v2 is Recommended (HIGH Confidence)
✅ 30%+ more signals than v1  
✅ Win rate within 5% of v1  
✅ Average return 10%+ higher  
✅ Better or equal max drawdown  
✅ Better Sharpe ratio  

**Action**: Consider adopting v2 enhancements for production

### When Results are NEUTRAL
⚠️ Mixed improvements across metrics  
⚠️ Statistical significance unclear  

**Action**: Run longer backtest period or monitor live performance

### When v1 is Better
❌ v2 generates fewer high-quality signals  
❌ Win rate significantly drops  
❌ Risk metrics worsen  

**Action**: Stick with v1 or refine v2 parameters

## Output Files

### Location
All results are saved in:
- `IN-testing/backtesting/results/` - Individual backtest results
- `IN-testing/backtesting/comparison_reports/` - Comparison reports

### Files Generated

#### Swing Trading
- `backtest_swing_v1_trades_*.csv` - All v1 swing trades
- `backtest_swing_v1_metrics_*.json` - v1 performance metrics
- `backtest_swing_v2_trades_*.csv` - All v2 swing trades
- `backtest_swing_v2_metrics_*.json` - v2 performance metrics
- `comparison_swing_*.html` - Interactive comparison report

#### BTST Trading
- `backtest_btst_v1_trades_*.csv` - All v1 BTST trades
- `backtest_btst_v1_metrics_*.json` - v1 performance metrics
- `backtest_btst_v2_trades_*.csv` - All v2 BTST trades
- `backtest_btst_v2_metrics_*.json` - v2 performance metrics
- `comparison_btst_*.html` - Interactive comparison report

## Advanced Options

### Skip Data Download
If you've already downloaded data:
```bash
./run_full_backtest.sh --test --skip-download
```

### Custom Date Range
```bash
./run_full_backtest.sh --start-date 2025-03-01 --end-date 2025-09-01
```

### Full Nifty 500 Analysis
```bash
./run_full_backtest.sh
# No --test flag = analyzes top 200 stocks
```

## Incremental Data Updates

The framework now supports **incremental updates** - only new data since the last run is downloaded!

### How It Works
- **First Run**: Downloads full 180 days of historical data
- **Subsequent Runs**: Only downloads data since last recorded date
- **Today's Data**: If data is already up-to-date (from today), skip download entirely

### Benefits
- **Faster**: Daily updates take seconds instead of minutes
- **Efficient**: No duplicate data, no wasted API calls
- **Automatic**: Rolling window maintenance (keeps latest 180 days)

### Usage
```bash
# First run - downloads all data
cd IN-testing/scripts
python data_fetcher.py --test

# Later runs - only downloads new data
python data_fetcher.py --test
# Skips stocks with today's data, incrementally updates others
```

### Force Full Re-download
```bash
python data_fetcher.py --test --force
```

## Results Tracking Database

All backtest comparisons are automatically saved to a **history database** for trend analysis!

### What Gets Tracked
- **Backtest Results**: v1 and v2 performance metrics for each run
- **Comparison Data**: Recommendations, confidence levels, deltas
- **Both Strategies**: Swing AND BTST results tracked separately
- **30-Day Trends**: View how recommendations change over time

### Database Location
`IN-testing/data/nifty500_historical.db`

### Tables
- `backtest_results_history` - Individual strategy performance over time
- `comparison_results_history` - Comparison results and recommendations
- `vw_30day_trend` - Last 30 days of trends (view)

### Automatic Saving
Results are saved automatically after running comparisons:
- `compare_swing_strategies.py` → Saves Swing results
- `compare_btst_strategies.py` → Saves BTST results

## Web Dashboard

View your backtest results history in an **interactive web dashboard**!

### Starting the Dashboard
```bash
cd IN-testing/backtesting
./start_dashboard.sh
```

Then open: **http://127.0.0.1:5001**

### Features
- **Overview Tab**: Current recommendations for both Swing and BTST
- **Strategy Tabs**: Detailed analysis for each strategy type
- **30-Day Trends**: Visual trend charts
- **Historical Results**: Full comparison history
- **Auto-Refresh**: Updates every 60 seconds
- **Real-Time**: See latest backtest results immediately

### Dashboard Sections

#### Overview
- Latest recommendations for Swing and BTST
- Confidence levels and criteria met
- Delta metrics (return, Sharpe, win rate)

#### Swing Strategy Tab
- Latest comparison results
- 30-day recommendation trend
- v1 vs v2 performance comparison
- Historical results table

#### BTST Strategy Tab
- Latest comparison results
- 30-day recommendation trend
- v1 vs v2 performance comparison
- Historical results table

### Stopping the Dashboard
Press `Ctrl+C` in the terminal

### Port Configuration
Default port is 5001 (different from AU dashboard on 5000). To change:
```bash
python view_results_dashboard.py --port 5002
```

## Troubleshooting

### "Virtual environment not found"
The script will use system Python. Ensure these packages are installed:
```bash
pip install pandas numpy yfinance scipy matplotlib plotly
```

### "Database not found"
Remove `--skip-download` flag to let the script download data first.

### "No trades generated"
- Date range may be too short
- Try different start/end dates
- Check that at least 30-50 trading days are in range

### Rate limit errors
The script has built-in rate limiting (60 calls/minute). If you still hit limits:
- Use `--test` mode first
- Wait 5-10 minutes between runs

## Technical Details

### Database
- **Location**: `IN-testing/data/nifty500_historical.db`
- **Type**: SQLite
- **Structure**: One table per stock + metadata + unified view
- **Size**: ~50-100 MB for 200 stocks

### Data Source
- **Provider**: Yahoo Finance (yfinance library)
- **Symbols**: Nifty 500 stocks with `.NS` suffix
- **Indicators**: EMA(20/50), SMA(200), RSI, MACD, ATR, Bollinger Bands, Volume Ratio, 52W high/low

### Strategy Versions

#### Swing v1 (Current)
- Market Cap ≥ ₹5,000 Cr, D/E ≤ 0.5, ROE ≥ 15%
- RSI: 40-60
- Volume ≥ 1.2x
- 5 entry signals
- Min score: 60/100
- Max results: 15

#### Swing v2 (Enhanced)
- Market Cap ≥ ₹5,000 Cr, D/E ≤ 0.8 (relaxed), ROE ≥ 12% (relaxed)
- RSI: 30-70 (wider)
- Volume ≥ 500k shares (liquidity)
- ATR ≥ 1.5% (volatility)
- 8 entry signals (added: Bollinger Bounce, Volume Surge, Consolidation Breakout)
- Min score: 55/100 (lowered)
- Max results: 20 (increased)

#### BTST v1 (Current)
- Day gain: 2-3%
- Volume ≥ 1.5x
- Close near high ≥ 90%
- Above 20 EMA
- Exclude IT, Pharma sectors
- Min score: 60/100
- Max results: 10

#### BTST v2 (Enhanced)
- Day gain: 1.5-4% (wider)
- Volume ≥ 1.3x (relaxed)
- Close near high ≥ 85% (relaxed)
- RSI < 70 (not overbought)
- MACD histogram > 0
- ATR ≥ 1.0%
- Min score: 55/100 (lowered)
- Max results: 15 (increased)

## Reference Implementation

This framework is based on the proven AU-testing framework for Australian stocks.
All patterns, optimizations, and best practices have been adapted for Indian markets:
- Parallel data fetching (5 workers, 25 stocks/batch)
- Thread-safe database operations
- Bulk inserts for performance
- Comprehensive technical indicators
- Statistical significance testing
- Interactive HTML reports

## Next Steps After Analysis

1. **Review HTML Reports**: Open in browser for interactive analysis
2. **Check Statistical Significance**: Look for p-values < 0.05
3. **Compare Entry Signals**: Which signals perform best?
4. **Analyze Sectors**: Which sectors benefit most from v2?
5. **Make Decision**: Adopt v2, stick with v1, or refine parameters

## Support

For issues or questions:
1. Check logs in `IN-testing/logs/`
2. Review error messages in terminal output
3. Ensure Python 3.8+ and required packages installed
4. Verify virtual environment is activated

---

**Framework Version**: 1.0  
**Last Updated**: 2025-11-11  
**Compatible With**: Indian Nifty 500 stocks, stockScannerIN venv  
**Tested On**: macOS, Python 3.9+
