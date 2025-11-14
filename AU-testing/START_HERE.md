# START HERE - ASX200 Strategy Backtesting

**Updated**: 2025-11-12

---

## What This Does

Tests your current swing trading strategy (v1) against an improved version (v2) using 6 months of ASX200 data, then tells you which one is better.

---

## ⚡ Quick Start (ONE Command)

```bash
cd AU-testing/backtesting
./run_full_backtest.sh --test
```

**What happens:**
1. Downloads 10 test stocks (~10 seconds with parallel download)
2. Runs v1 backtest (~30 seconds)
3. Runs v2 backtest (~30 seconds)
4. Generates comparison report
5. **Opens HTML report in your browser** ← THIS IS YOUR ANSWER

**Total time**: ~2 minutes

---

## What You'll See

The HTML report shows:

### Executive Summary
- **Recommendation**: Use v1 or v2? (with confidence level)
- **Key Metrics**: Side-by-side comparison

### Detailed Analysis
- Total trades (v2 should have 30-50% more)
- Win rate (should be within 5%)
- Average return (v2 target: 10-20% higher)
- Risk metrics (Sharpe ratio, max drawdown)
- Performance by sector
- Performance by entry signal type

### Decision

If v2 meets 4-5 success criteria:
- ✅ 30%+ more signals
- ✅ Win rate within 5% of v1
- ✅ Average return 10%+ higher
- ✅ Better Sharpe ratio
- ✅ Same/better max drawdown

→ **HIGH confidence: Adopt v2**

Otherwise: Stay with v1

---

## Full ASX200 (200 Stocks)

```bash
cd AU-testing/backtesting
./run_full_backtest.sh
```

**Time**: ~10 minutes total
- Data download: 5-7 minutes (parallel, optimized)
- v1 backtest: 1-2 minutes
- v2 backtest: 1-2 minutes
- Comparison: 30 seconds

---

## Files Created

After running, check:

```
AU-testing/backtesting/
├── results/
│   ├── backtest_v1_trades_*.csv       # v1 trade details
│   ├── backtest_v1_metrics_*.json     # v1 metrics
│   ├── backtest_v2_trades_*.csv       # v2 trade details
│   └── backtest_v2_metrics_*.json     # v2 metrics
└── comparison_reports/
    ├── comparison_report_*.html       # 🌟 OPEN THIS!
    ├── comparison_summary_*.txt       # Quick text summary
    └── comparison_metrics_*.csv       # Excel-ready
```

---

## What's Different: v1 vs v2

### v1 (Current)
- RSI: 35-65
- 52W High: 85-98%
- D/E ≤ 1.0, ROE ≥ 10%
- 5 entry signals
- Max 15 candidates
- Materials/Energy priority

### v2 (Enhanced)
- RSI: 30-70 (wider)
- 52W High: 80-100% (no cap)
- D/E ≤ 1.2, ROE ≥ 8% (relaxed)
- **NEW**: Volume ≥ 500k, ATR% ≥ 1.5%
- 8 entry signals (3 new)
- Max 20 candidates
- Healthcare/Consumer Staples priority (2025 market)

---

## Troubleshooting

### "No such file or directory"
```bash
# Make sure you're in the right directory
cd /Users/amitsaddi/Documents/git/IN-Stock-scanner/AU-testing/backtesting
./run_full_backtest.sh --test
```

### "Permission denied"
```bash
chmod +x run_full_backtest.sh
./run_full_backtest.sh --test
```

### "No module named pandas"
```bash
# Install dependencies (one-time)
cd /Users/amitsaddi/Documents/git/IN-Stock-scanner
source stockScannerIN/bin/activate
pip install -r AU-testing/requirements.txt
```

### Script runs but shows errors
```bash
# Check logs
ls -lt AU-testing/logs/
tail AU-testing/logs/data_fetcher_*.log
```

---

## That's It!

**One command gives you the answer**: Should you use v2 or stick with v1?

The HTML report has everything you need to make a data-driven decision.

---

## Advanced Options (Optional)

### Custom Date Range
```bash
./run_full_backtest.sh --start-date 2025-07-01 --end-date 2025-10-31
```

### Skip Download (if data already exists)
```bash
./run_full_backtest.sh --test --skip-download
```

### Full Command Options
```bash
./run_full_backtest.sh [--test] [--skip-download] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```

---

## What NOT to Do

- ❌ Don't run individual scripts (data_fetcher.py, backtest_v1.py, etc.) - use the master script
- ❌ Don't read all the documentation - just run the command
- ❌ Don't modify anything until you see the comparison report

---

## Incremental Data Updates

The data fetcher now downloads only NEW data since the last recorded date, making daily updates 90% faster.

### How It Works

**First Run (Full Download)**:
```bash
cd AU-testing/scripts
python3 data_fetcher.py --test --workers 5
# Downloads full 180 days (~2 minutes for 10 stocks)
```

**Subsequent Runs (Incremental)**:
```bash
# Next day
python3 data_fetcher.py --test --workers 5
# Downloads only 1 day of new data (~10 seconds)
```

### Key Features

- ✅ Skips stocks with today's data already downloaded
- ✅ Downloads only new dates for stocks that need updating
- ✅ Prevents duplicate dates automatically
- ✅ Maintains rolling 180-day window
- ✅ 90% faster for daily updates (8 minutes → 45 seconds for full ASX200)

### Force Full Re-download (if needed)

```bash
python3 data_fetcher.py --test --workers 5 --force
```

---

## Results Tracking Database

Every backtest run now automatically saves results to a history database for trend analysis.

### What Gets Tracked

**For Each Run**:
- Date of run (one record per day - latest run overwrites)
- Strategy type (swing)
- Version (v1 or v2)
- All metrics: total trades, win rate, avg return, Sharpe ratio, max drawdown
- Sector performance
- Entry signal distribution
- Comparison recommendation (use_v1, use_v2, or inconclusive)
- Confidence level (high, medium, low)

### Database Location

Results are stored in the same database as stock data:
```
AU-testing/data/asx200_historical.db
```

**Tables**:
- `backtest_results_history` - Individual v1/v2 run metrics
- `comparison_results_history` - Strategy recommendations
- `vw_30day_trend` - View for 30-day trend analysis

### Automatic Saving

Results are automatically saved when you run:
```bash
./run_full_backtest.sh --test
```

No manual intervention needed - just run your backtest and results are tracked!

---

## Web Dashboard

View your backtest results and trends through an interactive web dashboard.

### Starting the Dashboard

```bash
cd AU-testing/backtesting
./start_dashboard.sh
```

Then open your browser to: **http://localhost:5000**

### Dashboard Features

#### Overview Tab
- **Latest Recommendation**: Use v1 or v2? (with confidence level)
- **30-Day Summary**: How many times v2 won vs v1
- **Key Metrics**: Side-by-side performance comparison

#### Performance Trends
- Win rate over time (v1 vs v2)
- Average return trends
- Sharpe ratio comparison
- Trade volume trends

#### Historical Results Table
- Sortable table of all backtest runs
- Filter by date range
- Export to CSV

### Auto-Refresh

Dashboard automatically refreshes every 60 seconds to show latest results.

### Port Configuration

- AU Dashboard: **Port 5000**
- IN Dashboard: **Port 5001** (can run both simultaneously)

### Stopping the Dashboard

Press `Ctrl+C` in the terminal where the dashboard is running.

---

## Next Steps After Results

1. **Open the HTML report** (auto-opens or find in `comparison_reports/`)
2. **Read the recommendation** (top of report)
3. **Review the metrics** (side-by-side comparison)
4. **Make decision**: Adopt v2 or stay with v1
5. **If adopting v2**: Update production code in `src/markets/australia/`

---

**Questions? Just run the command and see what happens. The report explains everything.**
