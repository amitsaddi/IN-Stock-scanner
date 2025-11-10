#!/usr/bin/env python3
"""
Quick test script for batch fetching with just 2 symbols
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TESTING BATCH FETCH WITH 2 SYMBOLS")
print("=" * 60)
print()

from btst_scanner import BTSTScanner
from swing_scanner import SwingScanner

# Test with just 2 symbols
test_symbols = ['RELIANCE.NS', 'TCS.NS']

print("Testing BTST Scanner...")
print("-" * 60)
btst_scanner = BTSTScanner()
btst_candidates = btst_scanner.scan_for_btst(symbols=test_symbols, max_results=10)

print(f"\nFound {len(btst_candidates)} BTST candidates")
if btst_candidates:
    report = btst_scanner.generate_report(btst_candidates)
    print("\n" + report)
else:
    print("No BTST candidates found")

print("\n" + "=" * 60)
print("Testing Swing Scanner...")
print("-" * 60)
swing_scanner = SwingScanner()
swing_candidates = swing_scanner.scan_for_swing(symbols=test_symbols, max_results=10)

print(f"\nFound {len(swing_candidates)} swing candidates")
if swing_candidates:
    report = swing_scanner.generate_report(swing_candidates)
    print("\n" + report)
else:
    print("No swing candidates found")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
