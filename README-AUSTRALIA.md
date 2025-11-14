# Australian Stock Market Scanner - ASX 200

This module extends the Indian stock scanner to support the Australian Securities Exchange (ASX 200). It identifies swing trading opportunities in Australian stocks using technical analysis and fundamental filters.

## Stock Selection Workflow

### Australian Swing Scanner Workflow
```
┌─────────────────────────────────────────────────────────────┐
│  START: ASX 200 Stocks (200 stocks)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Batch Fetch Daily Data (100 days)                  │
│  • OHLCV data for all 200 stocks (.AX suffix)               │
│  • Takes ~1-2 minutes with batch fetching                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Calculate Technical Indicators (All Stocks)        │
│  • EMA(20), EMA(50), SMA(200)                               │
│  • RSI(14), MACD(12,26,9)                                   │
│  • Volume ratio vs 20-day average                           │
│  • 52-week high/low for momentum tracking                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Technical Filter                                   │
│  ✓ RSI: 35-65 (wider range for AU market)                   │
│  ✓ 52-week high proximity: 85-98%                           │
│  ✓ Price: Above EMA(20) preferred                           │
│  ✓ Volume: >1.2x average                                    │
│  ✓ MACD: Bullish or neutral                                 │
│  → Typically 40-80 stocks pass                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Fetch Fundamentals (Filtered Stocks Only)          │
│  • Market cap, Debt/Equity, ROE, PE ratio                    │
│  • Sector classification for rotation weighting              │
│  • Takes ~1-2 minutes with batching & rate limiting          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Fundamental Filter                                 │
│  ✓ Market Cap: >A$500M (50,000 lakhs)                       │
│  ✓ Debt/Equity: <1.0 (more lenient than India)             │
│  ✓ ROE: >10% (adjusted for AU market)                       │
│  → Typically 15-35 stocks pass                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Entry Signal Detection                             │
│  • Pullback (RSI 30-35, near support)                        │
│  • Breakout (price > EMA with volume)                        │
│  • MACD Cross (bullish crossover)                            │
│  • MA Cross (EMA 20 > EMA 50)                                │
│  • Trend Follow (above all MAs, near 52W high)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Apply Sector Rotation Weights                      │
│  • Materials: 1.2x (commodity strength)                      │
│  • Energy: 1.15x (resource economy)                          │
│  • Financials: 1.0x (standard)                               │
│  • IT/Healthcare: 0.8-0.9x (lower priority)                  │
│  • Multiply base score by sector weight                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: Score & Rank                                       │
│  • Technical score (35%): RSI, MACD, MA alignment            │
│  • Fundamental score (25%): ROE, D/E, Market Cap             │
│  • Momentum score (25%): 52W high proximity                  │
│  • Volume score (15%): Volume ratio                          │
│  • Final score = Base score × Sector weight                 │
│  • Sort by weighted score                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Top Swing Candidates (0-15 typically)              │
│  • Email report with sector-weighted analysis                │
│  • Telegram completion summary (always)                      │
│  • CSV: data/results/australia/swing_scan_YYYYMMDD.csv      │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **ASX 200 Coverage**: Scans top 200 Australian stocks by market capitalization
- **Swing Trading Focus**: Identifies medium-term trading opportunities (3-15 days)
- **Sector Rotation Weighting**: Applies sector-specific multipliers based on market conditions
- **52-Week High Tracking**: Monitors proximity to 52-week highs for momentum analysis
- **Automated Scheduling**: Runs daily after market close (4:15 PM AEST)

## Australian Market Criteria

### Fundamental Filters
- **Market Cap**: Minimum A$500M (50,000 lakhs)
- **Debt-to-Equity**: Maximum 1.0 (more lenient than Indian market)
- **ROE**: Minimum 10% (adjusted for Australian market)

### Technical Criteria
- **RSI Range**: 35-65 (wider range than Indian market)
- **52-Week High Proximity**: 85-98% (momentum indicator)
- **Volume**: Minimum 1.2x average
- **Trend**: Above 20/50 EMAs preferred
- **MACD**: Bullish crossover signals

### Sector Rotation Weights

The scanner applies sector-specific weights to favor sectors with strong momentum:

| Sector | Weight | Notes |
|--------|--------|-------|
| Materials | 1.2x | Strong commodity demand |
| Energy | 1.15x | Resource-driven economy |
| Financials | 1.0x | Banking sector strength |
| Consumer Discretionary | 1.0x | Standard weight |
| Industrials | 1.0x | Standard weight |
| Information Technology | 0.8x | Lower weight |
| Health Care | 0.9x | Moderate weight |
| Communication Services | 0.85x | Lower weight |

## Setup

### 1. Environment Variables

The scanner uses the same notification credentials as the Indian scanner:

```bash
# Email settings
EMAIL_FROM=your-email@gmail.com
EMAIL_TO=recipient@email.com
EMAIL_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Telegram settings
SEND_TELEGRAM=True
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

### 2. GitHub Actions Setup

The Australian scanner runs on a separate schedule from the Indian scanner:

- **Schedule**: Monday-Friday, 4:15 PM AEST (6:45 AM UTC)
- **Workflow**: `.github/workflows/daily_scan_australia.yml`
- **Command**: `python src/main_australia.py --type swing`

Enable the workflow:
1. Push the workflow file to your repository
2. Go to Actions tab in GitHub
3. Enable the "Daily Stock Scan - Australia" workflow

### 3. Manual Testing

Run locally to test before deployment:

```bash
# Test mode (scans only 10 stocks)
python src/main_australia.py --test

# Full scan (all ASX 200 stocks)
python src/main_australia.py
```

## File Structure

```
src/
├── config/
│   ├── australia_config.py       # Australian market configuration
│   └── ...
├── markets/
│   └── australia/
│       ├── swing_scanner.py       # Australian swing scanner
│       └── __init__.py
├── main_australia.py              # Entry point for Australian scanner
└── ...

data/
└── results/
    └── australia/
        └── swing_scan_YYYYMMDD.csv  # Scan results
```

## Output Format

Results are saved to `data/results/australia/swing_scan_YYYYMMDD.csv` with columns:

- Date, Symbol, Price, Sector, Entry_Type
- RSI, Volume_Ratio, 52W_High_%
- ROE_%, Debt_Equity
- Target, Stop_Loss, Score, Sector_Weight


## Strategy Notes

### Entry Signals
1. **Pullback**: RSI 30-35, near support
2. **Breakout**: Price > EMA_20 with volume
3. **MACD Cross**: Fresh bullish crossover
4. **MA Cross**: 20 EMA crosses above 50 EMA
5. **Trend Follow**: Above all MAs

### Risk Management
- **Target**: 12-15% based on entry type
- **Stop Loss**: 5-7% based on entry type
- **Position Sizing**: Consider sector weights
- **Hold Period**: 3-15 days typical

### Sector Rotation
- Materials and Energy sectors get priority (1.15-1.2x weight)
- Technology sector weighted lower (0.8x)
- Adjust weights quarterly based on market conditions

## Monitoring

### Telegram Notifications
- Scan completion summary (always sent)
- Detailed results (only when candidates found)
- Error alerts

### Email Reports
- Full analysis of all candidates
- Technical and fundamental metrics
- Entry/exit recommendations

## Troubleshooting

### No Candidates Found
- Market may be overbought (check RSI > 65)
- Volume too low across market
- Fundamentals not meeting criteria
- Adjust criteria in `australia_config.py` if needed

### API Rate Limits
- Scanner uses batch fetching to minimize API calls
- 2-second delay between batches
- Yahoo Finance free tier should be sufficient

### Timezone Issues
- Scanner uses `pytz` for Australian timezone
- Ensure `pytz` is installed: `pip install pytz`
- Workflow runs on UTC schedule (6:45 AM UTC = 4:15 PM AEST)

## Future Enhancements

Potential additions:
1. ASX 300 support (expand beyond top 200)
2. Intraday scanning support
3. Integration with Australian-specific data providers
4. Currency hedging analysis for multi-market portfolios
5. Correlation analysis between Indian and Australian markets

## Support

For issues specific to the Australian scanner:
1. Check logs in GitHub Actions
2. Verify ASX symbols use `.AX` suffix
3. Ensure market hours alignment (ASX: 10:00 AM - 4:00 PM AEST)
4. Review sector weights for current market conditions

## Disclaimer

This scanner is for educational and research purposes only. Always conduct your own research and consult with financial advisors before making investment decisions. Past performance does not guarantee future results.
