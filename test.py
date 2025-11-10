#!/usr/bin/env python3
"""
Quick test script - Run this to test the scanner locally
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("STOCK SCANNER - LOCAL TEST")
print("=" * 60)
print()

# Test imports
print("Testing imports...")
try:
    from config import Config
    from data_fetcher import DataFetcher
    from btst_scanner import BTSTScanner
    from swing_scanner import SwingScanner
    from notifier import Notifier
    print("✅ All imports successful\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test configuration
print("Testing configuration...")
try:
    config = Config()
    print(f"✅ Config loaded")
    print(f"   - Email enabled: {config.SEND_EMAIL}")
    print(f"   - Telegram enabled: {config.SEND_TELEGRAM}\n")
except Exception as e:
    print(f"❌ Config failed: {e}\n")

# Test data fetcher
print("Testing data fetcher...")
try:
    fetcher = DataFetcher()
    
    # Test with single stock
    data = fetcher.get_current_price_and_volume("RELIANCE.NS")
    if data:
        print(f"✅ Data fetch successful")
        print(f"   - Symbol: {data['symbol']}")
        print(f"   - Price: ₹{data['current_price']}")
        print(f"   - Change: {data['day_change_pct']:+.2f}%\n")
    else:
        print("⚠️  No data returned (market may be closed)\n")
except Exception as e:
    print(f"❌ Data fetch failed: {e}\n")

# Run test scan
print("Running BTST test scan (10 stocks)...")
print("This will take ~30 seconds...\n")

try:
    from btst_scanner import run_btst_scan
    
    candidates, report = run_btst_scan(test_mode=True)
    
    print(f"✅ BTST scan complete: {len(candidates)} candidates found\n")
    
    if candidates:
        print("Top 3 candidates:")
        for i, c in enumerate(candidates[:3], 1):
            print(f"{i}. {c.symbol} - ₹{c.current_price} ({c.day_change_pct:+.1f}%) - Score: {c.score:.0f}")
    else:
        print("No candidates found (normal if market is closed or no setups)")
    
    print()

except Exception as e:
    print(f"❌ BTST scan failed: {e}\n")

# Test notifications (optional)
print("=" * 60)
print("To test notifications, uncomment the code in this script")
print("=" * 60)

# Uncomment below to test notifications
# print("\nTesting notifications...")
# try:
#     notifier = Notifier()
#     
#     test_msg = "TEST: Stock scanner is working!"
#     
#     if config.SEND_EMAIL:
#         notifier.send_email("Test - Stock Scanner", test_msg)
#         print("✅ Test email sent")
#     
#     if config.SEND_TELEGRAM:
#         notifier.send_telegram(test_msg)
#         print("✅ Test Telegram sent")
# 
# except Exception as e:
#     print(f"❌ Notification failed: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("1. Check results in data/results/ folder")
print("2. Review the report above")
print("3. If all looks good, deploy to GitHub Actions")
print("\nTo run full scan: python src/main.py --type both")
