# 📊 Stock Scanner - Automated BTST & Swing Trading Scanner

Automated Python scanner that runs daily via GitHub Actions to identify BTST and swing trading opportunities in Indian stock market (Nifty 500).

## Quick Links
- **Indian Market**: See main documentation below
- **Australian Market**: See [README-AUSTRALIA.md](README-AUSTRALIA.md) for ASX-specific documentation

## Stock Selection Workflow

### BTST Scanner Workflow
```
┌─────────────────────────────────────────────────────────────┐
│  START: Nifty 500 Stocks (503 stocks)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Batch Fetch Current Data (yfinance)                │
│  • Current price, volume, day's OHLC                         │
│  • Previous close                                            │
│  • Takes ~30 seconds for all 503 stocks                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Quick Filter - Price Action                        │
│  ✓ Day gain: 2.0% - 3.5%                                    │
│  ✓ Near day high: >90%                                       │
│  ✓ Exclude: IT, Pharma sectors                              │
│  → Typically 10-30 stocks pass                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Fetch Historical Data (Filtered Stocks Only)       │
│  • Last 100 days OHLCV data                                  │
│  • Calculate: EMA(20), Volume avg(20)                        │
│  • Takes ~1-2 minutes for filtered stocks                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Technical Filter                                   │
│  ✓ Volume: >1.5x average                                    │
│  ✓ Price: Above EMA(20)                                     │
│  ✓ Momentum: Sustained intraday strength                    │
│  → Typically 3-10 stocks pass                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Score & Rank                                       │
│  • Price momentum score (40%)                                │
│  • Volume score (30%)                                        │
│  • High proximity score (30%)                                │
│  • Sort by total score                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Top BTST Candidates (0-10 typically)               │
│  • Email report with entry/exit targets                      │
│  • Telegram completion summary (always)                      │
│  • CSV save: data/results/india/btst_scan_YYYYMMDD.csv      │
└─────────────────────────────────────────────────────────────┘
```

### Swing Scanner Workflow
```
┌─────────────────────────────────────────────────────────────┐
│  START: Nifty 500 Stocks (503 stocks)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Batch Fetch Daily Data (100 days)                  │
│  • OHLCV data for all 503 stocks                             │
│  • Takes ~1-2 minutes with batch fetching                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Calculate Technical Indicators (All Stocks)        │
│  • EMA(20), EMA(50), SMA(200)                               │
│  • RSI(14), MACD(12,26,9)                                   │
│  • Volume ratio vs 20-day average                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Technical Filter                                   │
│  ✓ RSI: 40-60 (neutral zone)                                │
│  ✓ Price: Above EMA(20), EMA(50), SMA(200)                  │
│  ✓ MACD: Bullish or neutral                                 │
│  ✓ Volume: >1.3x average                                    │
│  → Typically 150-250 stocks pass                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Fetch Fundamentals (Filtered Stocks Only)          │
│  • Market cap, Debt/Equity, ROE, PE ratio                    │
│  • Sector classification                                     │
│  • Takes ~2-3 minutes with batching & rate limiting          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Fundamental Filter                                 │
│  ✓ Market Cap: >₹5,000 Cr                                   │
│  ✓ Debt/Equity: <0.5                                        │
│  ✓ ROE: >15%                                                │
│  ✓ Sector: Prefer Defence, Capital Goods, Infrastructure    │
│  → Typically 5-15 stocks pass                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Entry Signal Detection                             │
│  • Pullback (RSI 40-45, near support)                        │
│  • Breakout (price > EMA with volume)                        │
│  • MACD Cross (bullish crossover)                            │
│  • MA Cross (EMA 20 > EMA 50)                                │
│  • Trend Follow (above all MAs)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Score & Rank                                       │
│  • Technical score (40%): RSI, MACD, MA alignment            │
│  • Fundamental score (30%): ROE, D/E, Market Cap             │
│  • Volume score (20%): Volume ratio                          │
│  • Sector bonus (10%): Preferred sectors get +15 points     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Top Swing Candidates (0-15 typically)              │
│  • Email report with detailed analysis                       │
│  • Telegram completion summary (always)                      │
│  • CSV save: data/results/india/swing_scan_YYYYMMDD.csv     │
└─────────────────────────────────────────────────────────────┘
```

## Features (India)

- **BTST Scanner**: Identifies buy-today-sell-tomorrow opportunities based on late-day momentum
- **Swing Scanner**: Finds swing trading setups (3-15 day holds) using technical + fundamental analysis
- **Automated Execution**: Runs daily at 3:15 PM IST via GitHub Actions
- **Multi-Channel Notifications**: Email and Telegram alerts with smart completion summaries
- **Batch Data Fetching**: Optimized to scan 500+ stocks in ~3-5 minutes (5-10x faster)
- **Historical Tracking**: Saves all scan results to CSV
- **Configurable Criteria**: Easy to customize scanning parameters
- **Always-On Monitoring**: Telegram notifications even when no opportunities found

## Quick Start

### 1. Fork/Clone This Repository

```bash
git clone https://github.com/yourusername/stock-scanner.git
cd stock-scanner
```

### 2. Set Up GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:

**Email Notifications:**
- `EMAIL_FROM`: Your Gmail address
- `EMAIL_TO`: Recipient email address
- `EMAIL_PASSWORD`: Gmail app password (not your regular password!)
- `SMTP_SERVER`: `smtp.gmail.com`
- `SMTP_PORT`: `587`
- `SEND_EMAIL`: `True`

**Telegram Notifications (Recommended):**
- `TELEGRAM_BOT_TOKEN`: Your bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Your chat ID
- `SEND_TELEGRAM`: `True` (recommended for daily completion notifications)

### 3. Enable GitHub Actions

- Go to Actions tab in your repository
- Click "I understand my workflows, go ahead and enable them"

### 4. Manual Test Run

- Go to Actions → Daily Stock Scanner
- Click "Run workflow"
- Select scan type and enable test mode
- Click "Run workflow"

## How to Get Gmail App Password

1. Go to Google Account → Security
2. Enable 2-Step Verification
3. Go to App passwords
4. Select "Mail" and "Other (Custom name)"
5. Generate and copy the 16-character password
6. Use this password in `EMAIL_PASSWORD` secret

## How to Set Up Telegram Bot

1. Open Telegram and search for @BotFather
2. Send `/newbot` and follow instructions
3. Copy the bot token (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. Start a chat with your bot
5. Get your chat ID:
   - Send a message to your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your `chat.id` in the response

## Project Structure

```
stock-scanner/
├── .github/
│   └── workflows/
│       ├── daily_scan_india.yml       # Indian market workflow
│       └── daily_scan_australia.yml   # Australian market workflow
├── src/
│   ├── config/
│   │   ├── base_config.py            # Shared configuration
│   │   ├── india_config.py           # India-specific config
│   │   └── australia_config.py       # Australia-specific config
│   ├── shared/
│   │   ├── data_fetcher.py           # Multi-market data fetching
│   │   └── notifier.py               # Email/Telegram notifications
│   ├── markets/
│   │   ├── india/
│   │   │   ├── btst_scanner.py       # BTST scanner (India only)
│   │   │   └── swing_scanner.py      # Swing scanner (India)
│   │   └── australia/
│   │       └── swing_scanner.py      # Swing scanner (Australia)
│   ├── main_india.py                 # Indian market runner
│   ├── main_australia.py             # Australian market runner
│   └── main.py                       # Backwards compatibility (deprecated)
├── data/
│   └── results/
│       ├── india/                    # Indian scan results
│       └── australia/                # Australian scan results
├── requirements.txt                  # Python dependencies
├── README.md                         # Indian market documentation
└── README-AUSTRALIA.md               # Australian market documentation
```

## Local Development

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Set Environment Variables

Create `.env` file:

```bash
# Email
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=recipient@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SEND_EMAIL=True

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SEND_TELEGRAM=False
```

### Run Locally

```bash
# Test run (only 10 stocks)
python src/main_india.py --test

# Full BTST scan
python src/main_india.py --type btst

# Full swing scan
python src/main_india.py --type swing

# Both scans
python src/main_india.py --type both
```

## Customization

### Adjust Scanning Criteria

Edit `src/config.py`:

```python
@dataclass
class BTSTCriteria:
    min_gain_percent: float = 2.0      # Minimum day gain
    max_gain_percent: float = 3.5      # Maximum day gain
    min_volume_multiplier: float = 1.5 # Volume vs average
    # ... more parameters

@dataclass
class SwingCriteria:
    min_market_cap: float = 5000       # Min market cap (crores)
    max_debt_to_equity: float = 0.5    # Max debt/equity ratio
    min_roe: float = 15.0              # Min ROE %
    # ... more parameters
```

### Change Schedule

Edit `.github/workflows/daily_scan.yml`:

```yaml
schedule:
  # Run at 3:15 PM IST (9:45 AM UTC)
  - cron: '45 9 * * 1-5'  # Mon-Fri
```

## Scan Criteria

### BTST Criteria
- ✅ 2-3% gain at 3:00 PM
- ✅ Volume 1.5x above average
- ✅ Closing near day high (>90%)
- ✅ Above 20 EMA
- ✅ Liquid stocks (Nifty 500)
- ❌ Exclude IT, Pharma sectors

### Swing Criteria
- ✅ Market cap > ₹5,000 Cr
- ✅ Debt/Equity < 0.5
- ✅ ROE > 15%
- ✅ RSI 40-60
- ✅ Above 20/50/200 MAs
- ✅ Volume confirmation
- ✅ Preferred sectors: Defence, Capital Goods, Infrastructure

## Output & Notifications

### Smart Notification System

**Telegram** (Always sent):
- Completion summary with scan results
- Sent even when no opportunities found
- Quick status check on your phone

**Email** (Only when opportunities found):
- Detailed analysis reports
- Full stock recommendations
- Entry/exit targets and stop losses

### Telegram Completion Summary Example

```
📊 Stock Scanner - Scan Complete
10 Nov 2025, 03:15 PM IST

🟢 BTST Scan: 3 opportunities found
🟠 Swing Scan: 5 stocks found

✅ Detailed reports sent via email.
```

**Or when no stocks found:**

```
📊 Stock Scanner - Scan Complete
10 Nov 2025, 03:15 PM IST

🟢 BTST Scan: 0 opportunities found
🟠 Swing Scan: 0 stocks found

ℹ️ No trading opportunities found today.
The scanner ran successfully but market conditions didn't meet our criteria.
```

### Email Report Example

```
🟢 BTST OPPORTUNITIES - 07 November 2025, 03:15 PM
==================================================

1. RELIANCE
   Price: ₹1481.70 | Gain: +2.3%
   Volume: 1.8x | Near High: 95%
   Sector: Energy | Score: 85/100
   Entry: Buy at 3:00-3:20 PM | Target: ₹1511 (2%)
   Stop Loss: ₹1452 (2%)

2. TCS
   ...
```

### CSV Output

Results saved to `data/results/btst_scan_YYYYMMDD.csv` and `swing_scan_YYYYMMDD.csv`

## Troubleshooting

### Scanner Not Running
- Check if GitHub Actions is enabled
- Verify cron schedule is correct for your timezone
- Check Actions tab for error logs

### Not Receiving Notifications
- Verify secrets are set correctly (no typos)
- For Gmail: Make sure app password is used (not regular password)
- For Telegram: Test bot responds when you message it
- Check spam folder for emails

### Python Errors
- Make sure all dependencies in requirements.txt
- Check if yfinance API is working (sometimes rate limited)
- Review logs in GitHub Actions

## Performance & Optimization

### Recent Improvements (v2.0)

- ✅ **Batch Data Fetching**: Refactored to use `yf.download()` for multiple stocks simultaneously
- ✅ **5-10x Speed Improvement**: Scans 500+ stocks in ~3-5 minutes (down from 25+ minutes)
- ✅ **Smart Filtering**: Fetches detailed data only for stocks that pass initial criteria
- ✅ **Error Suppression**: Clean logs without yfinance error spam
- ✅ **Always-On Notifications**: Telegram completion summary every run

### Scan Performance

| Scanner | Stocks Scanned | Time (Approx) |
|---------|---------------|---------------|
| BTST    | 503 (Nifty 500) | ~2-3 minutes |
| Swing   | 503 → ~200 filtered | ~3-5 minutes |
| **Total** | **503 stocks** | **~5-8 minutes** |

## Limitations

- **Data Source**: Uses Yahoo Finance (yfinance) - free but occasionally rate limited
- **Market Hours**: Best results when run during market hours (9:15 AM - 3:30 PM IST)
- **Historical Data**: Limited to what yfinance provides
- **Batch Optimization**: Uses intelligent batching to avoid rate limits

## Technical Architecture

### Data Fetching Strategy

**Batch Operations** (New in v2.0):
```python
# Old: Sequential fetching (slow)
for symbol in symbols:
    data = fetch_single(symbol)
    time.sleep(rate_limit)

# New: Batch fetching (5-10x faster)
all_data = yf.download(symbols, threads=True)
filtered = [s for s in symbols if meets_criteria(s)]
detailed_data = fetch_detailed(filtered)
```

**Smart Filtering**:
1. Fetch current prices for all 503 stocks (batch)
2. Filter by gain criteria (e.g., 2%+ for BTST)
3. Fetch historical data only for filtered stocks
4. Fetch fundamentals only for passing candidates

### Test Script

Quick test with 2 symbols:
```bash
python test_batch.py
```

## Future Enhancements

- [x] ~~Batch data fetching~~ (Completed in v2.0)
- [x] ~~Smart notification system~~ (Completed in v2.0)
- [ ] Add NSEpy as backup data source
- [ ] Implement backtesting module
- [ ] Add performance tracking
- [ ] Create web dashboard for results
- [ ] Add more technical indicators
- [ ] Support for F&O analysis

## Disclaimer

⚠️ **This tool is for educational purposes only.**

- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research
- Trade at your own risk
- The creator is not responsible for any trading losses

## License

MIT License - Feel free to modify and use

## Contributing

Pull requests welcome! Please:
1. Fork the repo
2. Create feature branch
3. Test your changes
4. Submit PR with description

## Support

Issues? Questions?
- Open a GitHub issue
- Check existing issues first
- Provide error logs if reporting bugs

---

Built with ❤️ for Indian stock market traders

**Remember**: The best trade is the one you don't take if you're uncertain!
