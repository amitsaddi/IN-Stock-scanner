# 🚀 Quick Setup Guide - Stock Scanner

## Step-by-Step GitHub Deployment

### 1. Create GitHub Repository (5 minutes)

1. Go to https://github.com/new
2. Name: `stock-scanner`
3. Make it **Private**
4. Don't initialize with README
5. Click "Create repository"

### 2. Upload Code to GitHub

```bash
cd stock-scanner-project
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-scanner.git
git push -u origin main
```

### 3. Set Up Gmail App Password (5 minutes)

1. Go to https://myaccount.google.com/security
2. Enable "2-Step Verification"
3. Search for "App passwords"
4. Select "Mail" → "Other (Custom name)" → Type "Stock Scanner"
5. Click "Generate"
6. **Copy the 16-character password** (you'll need this)

### 4. Set Up GitHub Secrets (5 minutes)

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click "New repository secret"
3. Add these secrets one by one:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `EMAIL_FROM` | Your Gmail | `your.email@gmail.com` |
| `EMAIL_TO` | Where to send alerts | `your.email@gmail.com` |
| `EMAIL_PASSWORD` | Gmail app password | `abcd efgh ijkl mnop` |
| `SMTP_SERVER` | Gmail SMTP | `smtp.gmail.com` |
| `SMTP_PORT` | Port | `587` |
| `SEND_EMAIL` | Enable email | `True` |
| `SEND_TELEGRAM` | Enable Telegram | `False` |

### 5. Enable GitHub Actions (2 minutes)

1. Go to **Actions** tab in your repo
2. Click "I understand my workflows, go ahead and enable them"

### 6. Test Run (2 minutes)

1. Go to **Actions** → **Daily Stock Scanner**
2. Click "Run workflow" dropdown
3. Select:
   - Scan type: `both`
   - Test mode: ✅ **Check this** (will scan only 10 stocks)
4. Click "Run workflow"
5. Wait 2-3 minutes
6. Check your email!

---

## Schedule

The scanner runs automatically **Monday-Friday at 3:15 PM IST**.

You can also run manually anytime from Actions tab.

---

## Optional: Set Up Telegram (10 minutes)

### Get Bot Token

1. Open Telegram
2. Search for `@BotFather`
3. Send `/newbot`
4. Follow instructions
5. Copy the token (looks like `123456:ABC-DEF1234...`)

### Get Chat ID

1. Send a message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find `"chat":{"id":123456789}` in the response
4. Copy the number

### Add to GitHub Secrets

- `TELEGRAM_BOT_TOKEN`: Your bot token
- `TELEGRAM_CHAT_ID`: Your chat ID
- `SEND_TELEGRAM`: Change to `True`

---

## Troubleshooting

### ❌ "No module named 'config'"

**Fix**: Make sure you're in the `src/` directory when running locally:
```bash
cd src
python main.py --test
```

### ❌ Email not received

**Checks**:
- App password correct? (16 chars, no spaces in secret)
- Spam folder?
- 2-Step verification enabled?
- Secret names exactly match (case-sensitive)

### ❌ GitHub Actions failing

**Checks**:
- All secrets set?
- Actions enabled?
- Check error logs in Actions tab

### ❌ "Authentication failed" for Gmail

**Fix**: 
- You MUST use app password, not your regular Gmail password
- Make sure 2-Step verification is ON first

---

## Daily Workflow

1. **3:15 PM IST** - Scanner runs automatically
2. **3:20 PM IST** - You receive email/Telegram
3. **Review candidates** - Check the report
4. **For BTST** - Enter before 3:25 PM if suitable
5. **Next morning** - Exit BTST positions in first 15 min

---

## Files to Customize

### Want to change criteria?

Edit `src/config.py`:
- BTST min/max gain percent
- Swing min ROE, debt/equity
- RSI ranges
- Volume multipliers

### Want to change schedule?

Edit `.github/workflows/daily_scan.yml`:
```yaml
schedule:
  - cron: '45 9 * * 1-5'  # 3:15 PM IST = 9:45 AM UTC
```

---

## Cost

**$0/month** - Completely free:
- GitHub Actions: 2000 min/month free
- This scanner uses ~10-15 min/day
- Gmail: Free
- Telegram: Free

---

## What You Get

### BTST Email (Daily at 3:15 PM)
```
🟢 BTST OPPORTUNITIES - 07 Nov 2025, 03:15 PM

1. RELIANCE
   Price: ₹1481 | Gain: +2.3%
   Volume: 1.8x | Near High: 95%
   Score: 85/100
   Target: ₹1511 | SL: ₹1452

2. TCS
   ...
```

### Swing Email (Daily at 3:15 PM)
```
🟠 SWING TRADING WATCHLIST - 07 Nov 2025

1. HAL - PULLBACK
   Price: ₹4200 | RSI: 45 | ROE: 28%
   Target: ₹4704 (12%) | SL: ₹3948
   Score: 78/100

2. MAZAGON
   ...
```

---

## Pro Tips

1. **First week**: Run in test mode daily to validate
2. **Second week**: Enable full scan
3. **Track results**: Check CSV files in `data/results/`
4. **Review weekly**: See what worked, adjust criteria
5. **Don't blindly follow**: Scanner finds candidates, YOU decide

---

## Support

Issues? Open a GitHub issue with:
- Error message
- Screenshot of Actions log
- What you expected vs what happened

---

**You're all set! 🎉**

The scanner will now run automatically every trading day at 3:15 PM IST.

Good luck with your trading! 📈
