#!/bin/bash

################################################################################
# Nifty 500 Strategy Backtesting - Full Workflow
#
# This script runs the complete backtesting workflow:
# 1. Initialize database (if needed)
# 2. Download historical data (if needed)
# 3. Run v1 strategy backtest
# 4. Run v2 strategy backtest
# 5. Generate comparison reports
#
# Usage:
#   ./run_full_backtest.sh [--test] [--skip-download] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
#
# Options:
#   --test              Run in test mode (10 stocks only)
#   --skip-download     Skip data download step
#   --start-date        Backtest start date (default: 2025-05-01)
#   --end-date          Backtest end date (default: 2025-11-01)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Virtual environment path (relative to project root)
VENV_PATH="../../stockScannerIN"

# Default parameters
TEST_MODE=false
SKIP_DOWNLOAD=false
START_DATE="2025-05-01"
END_DATE="2025-11-01"
DB_PATH="../data/nifty500_historical.db"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            DB_PATH="../data/test.db"
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_full_backtest.sh [--test] [--skip-download] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}    Nifty 500 Strategy Backtesting - Full Workflow${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Test Mode:      ${YELLOW}$TEST_MODE${NC}"
echo -e "  Skip Download:  ${YELLOW}$SKIP_DOWNLOAD${NC}"
echo -e "  Date Range:     ${YELLOW}$START_DATE to $END_DATE${NC}"
echo -e "  Database:       ${YELLOW}$DB_PATH${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "run_full_backtest.sh" ]; then
    echo -e "${RED}Error: Please run this script from IN-testing/backtesting directory${NC}"
    exit 1
fi

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment: $VENV_PATH${NC}"
    source "$VENV_PATH/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo ""
elif [ -f "$VENV_PATH/Scripts/activate" ]; then
    # Windows Git Bash
    echo -e "${GREEN}Activating virtual environment: $VENV_PATH${NC}"
    source "$VENV_PATH/Scripts/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo ""
else
    echo -e "${YELLOW}Warning: Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Proceeding with system Python...${NC}"
    echo ""
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Display Python version
echo -e "${GREEN}Using Python: $(which python3)${NC}"
echo -e "${GREEN}Python version: $(python3 --version)${NC}"
echo ""

# Create necessary directories
mkdir -p ../logs
mkdir -p ./results
mkdir -p ./comparison_reports

################################################################################
# Step 1: Download Historical Data (if needed)
################################################################################

if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}Step 1/8: Downloading Historical Data${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    cd ../scripts

    if [ "$TEST_MODE" = true ]; then
        echo -e "${YELLOW}Running in TEST MODE (10 stocks with parallel download)...${NC}"
        python3 data_fetcher.py --test --db-path "$DB_PATH" --workers 5
    else
        echo -e "${YELLOW}Downloading full Nifty 500 data with parallel download (5-10 minutes)...${NC}"
        python3 data_fetcher.py --db-path "$DB_PATH" --workers 5
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Data download completed successfully${NC}"
    else
        echo -e "${RED}✗ Data download failed${NC}"
        exit 1
    fi

    cd ../backtesting
    echo ""
else
    echo -e "${YELLOW}Skipping data download step (using existing data)${NC}"
    echo ""
fi

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}Error: Database not found at $DB_PATH${NC}"
    echo -e "${RED}Run without --skip-download to create database${NC}"
    exit 1
fi

################################################################################
# Step 2: Run v1 Strategy Backtest
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 2/8: Running v1 (Current) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 swing_backtest_v1.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --db-path "$DB_PATH" \
    --output-dir ./results \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ v1 backtest completed successfully${NC}"
else
    echo -e "${RED}✗ v1 backtest failed${NC}"
    exit 1
fi

# Find the most recent v1 results
V1_METRICS=$(ls -t results/swing_backtest_v1_metrics_*.json | head -1)
V1_TRADES=$(ls -t results/swing_backtest_v1_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $V1_METRICS${NC}"
echo -e "${GREEN}  Trades:  $V1_TRADES${NC}"
echo ""

################################################################################
# Step 3: Run v2 Strategy Backtest
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 3/8: Running v2 (Enhanced) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 swing_backtest_v2.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --db-path "$DB_PATH" \
    --output-dir ./results \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ v2 backtest completed successfully${NC}"
else
    echo -e "${RED}✗ v2 backtest failed${NC}"
    exit 1
fi

# Find the most recent v2 results
V2_METRICS=$(ls -t results/swing_backtest_v2_metrics_*.json | head -1)
V2_TRADES=$(ls -t results/swing_backtest_v2_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $V2_METRICS${NC}"
echo -e "${GREEN}  Trades:  $V2_TRADES${NC}"
echo ""

################################################################################
# Step 4: Run BTST v1 Strategy Backtest
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 4/8: Running BTST v1 (Current) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 btst_backtest_v1.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --db-path "$DB_PATH" \
    --output-dir ./results \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BTST v1 backtest completed successfully${NC}"
else
    echo -e "${RED}✗ BTST v1 backtest failed${NC}"
    exit 1
fi

# Find the most recent BTST v1 results
BTST_V1_METRICS=$(ls -t results/btst_backtest_v1_metrics_*.json | head -1)
BTST_V1_TRADES=$(ls -t results/btst_backtest_v1_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $BTST_V1_METRICS${NC}"
echo -e "${GREEN}  Trades:  $BTST_V1_TRADES${NC}"
echo ""

################################################################################
# Step 5: Run BTST v2 Strategy Backtest
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 5/8: Running BTST v2 (Enhanced) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 btst_backtest_v2.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --db-path "$DB_PATH" \
    --output-dir ./results \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BTST v2 backtest completed successfully${NC}"
else
    echo -e "${RED}✗ BTST v2 backtest failed${NC}"
    exit 1
fi

# Find the most recent BTST v2 results
BTST_V2_METRICS=$(ls -t results/btst_backtest_v2_metrics_*.json | head -1)
BTST_V2_TRADES=$(ls -t results/btst_backtest_v2_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $BTST_V2_METRICS${NC}"
echo -e "${GREEN}  Trades:  $BTST_V2_TRADES${NC}"
echo ""

################################################################################
# Step 6: Generate Swing Comparison Reports
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 6/8: Generating Swing Comparison Reports${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 compare_swing_strategies.py \
    --v1-metrics "$V1_METRICS" \
    --v1-trades "$V1_TRADES" \
    --v2-metrics "$V2_METRICS" \
    --v2-trades "$V2_TRADES" \
    --output-dir ./comparison_reports \
    --format all

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Swing comparison reports generated successfully${NC}"
else
    echo -e "${RED}✗ Swing comparison report generation failed${NC}"
    exit 1
fi

# Find the most recent Swing comparison reports
SWING_COMPARISON_HTML=$(ls -t comparison_reports/swing_comparison_report_*.html | head -1)
SWING_COMPARISON_TXT=$(ls -t comparison_reports/swing_comparison_summary_*.txt | head -1)
SWING_COMPARISON_CSV=$(ls -t comparison_reports/swing_comparison_metrics_*.csv | head -1)

echo -e "${GREEN}  HTML:    $SWING_COMPARISON_HTML${NC}"
echo -e "${GREEN}  Summary: $SWING_COMPARISON_TXT${NC}"
echo -e "${GREEN}  CSV:     $SWING_COMPARISON_CSV${NC}"
echo ""

################################################################################
# Step 7: Generate BTST Comparison Reports
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 7/8: Generating BTST Comparison Reports${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 compare_btst_strategies.py \
    --v1-metrics "$BTST_V1_METRICS" \
    --v1-trades "$BTST_V1_TRADES" \
    --v2-metrics "$BTST_V2_METRICS" \
    --v2-trades "$BTST_V2_TRADES" \
    --output-dir ./comparison_reports \
    --format all

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BTST comparison reports generated successfully${NC}"
else
    echo -e "${RED}✗ BTST comparison report generation failed${NC}"
    exit 1
fi

# Find the most recent BTST comparison reports
BTST_COMPARISON_HTML=$(ls -t comparison_reports/btst_comparison_report_*.html | head -1)
BTST_COMPARISON_TXT=$(ls -t comparison_reports/btst_comparison_summary_*.txt | head -1)
BTST_COMPARISON_CSV=$(ls -t comparison_reports/btst_comparison_metrics_*.csv | head -1)

echo -e "${GREEN}  HTML:    $BTST_COMPARISON_HTML${NC}"
echo -e "${GREEN}  Summary: $BTST_COMPARISON_TXT${NC}"
echo -e "${GREEN}  CSV:     $BTST_COMPARISON_CSV${NC}"
echo ""

################################################################################
# Step 8: Display Executive Summaries
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 8/8: Executive Summaries${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

echo -e "${CYAN}=== SWING TRADING COMPARISON ===${NC}"
echo ""
if [ -f "$SWING_COMPARISON_TXT" ]; then
    cat "$SWING_COMPARISON_TXT"
else
    echo -e "${YELLOW}Swing summary file not found${NC}"
fi

echo ""
echo ""
echo -e "${CYAN}=== BTST TRADING COMPARISON ===${NC}"
echo ""
if [ -f "$BTST_COMPARISON_TXT" ]; then
    cat "$BTST_COMPARISON_TXT"
else
    echo -e "${YELLOW}BTST summary file not found${NC}"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}✓ Complete Workflow Finished Successfully!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Display file locations
echo -e "${GREEN}Results Available At:${NC}"
echo ""
echo -e "${YELLOW}Swing v1 Backtest:${NC}"
echo -e "  • Trades:  $V1_TRADES"
echo -e "  • Metrics: $V1_METRICS"
echo ""
echo -e "${YELLOW}Swing v2 Backtest:${NC}"
echo -e "  • Trades:  $V2_TRADES"
echo -e "  • Metrics: $V2_METRICS"
echo ""
echo -e "${YELLOW}BTST v1 Backtest:${NC}"
echo -e "  • Trades:  $BTST_V1_TRADES"
echo -e "  • Metrics: $BTST_V1_METRICS"
echo ""
echo -e "${YELLOW}BTST v2 Backtest:${NC}"
echo -e "  • Trades:  $BTST_V2_TRADES"
echo -e "  • Metrics: $BTST_V2_METRICS"
echo ""
echo -e "${YELLOW}Swing Comparison:${NC}"
echo -e "  • HTML:    $SWING_COMPARISON_HTML"
echo -e "  • Summary: $SWING_COMPARISON_TXT"
echo -e "  • CSV:     $SWING_COMPARISON_CSV"
echo ""
echo -e "${YELLOW}BTST Comparison:${NC}"
echo -e "  • HTML:    $BTST_COMPARISON_HTML"
echo -e "  • Summary: $BTST_COMPARISON_TXT"
echo -e "  • CSV:     $BTST_COMPARISON_CSV"
echo ""

# Open HTML reports (if available)
if command -v open &> /dev/null; then
    if [ -f "$SWING_COMPARISON_HTML" ]; then
        echo -e "${GREEN}Opening Swing comparison report in browser...${NC}"
        open "$SWING_COMPARISON_HTML"
    fi
    if [ -f "$BTST_COMPARISON_HTML" ]; then
        echo -e "${GREEN}Opening BTST comparison report in browser...${NC}"
        open "$BTST_COMPARISON_HTML"
    fi
elif command -v xdg-open &> /dev/null; then
    if [ -f "$SWING_COMPARISON_HTML" ]; then
        echo -e "${GREEN}Opening Swing comparison report in browser...${NC}"
        xdg-open "$SWING_COMPARISON_HTML"
    fi
    if [ -f "$BTST_COMPARISON_HTML" ]; then
        echo -e "${GREEN}Opening BTST comparison report in browser...${NC}"
        xdg-open "$BTST_COMPARISON_HTML"
    fi
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Thank you for using the Nifty 500 Backtesting Framework!${NC}"
echo -e "${BLUE}================================================================${NC}"
