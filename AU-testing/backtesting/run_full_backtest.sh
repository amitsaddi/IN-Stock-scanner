#!/bin/bash

################################################################################
# ASX200 Strategy Backtesting - Full Workflow
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
DB_PATH="../data/asx200_historical.db"

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
echo -e "${BLUE}    ASX200 Strategy Backtesting - Full Workflow${NC}"
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
    echo -e "${RED}Error: Please run this script from AU-testing/backtesting directory${NC}"
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
    echo -e "${BLUE}Step 1/5: Downloading Historical Data${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    cd ../scripts

    if [ "$TEST_MODE" = true ]; then
        echo -e "${YELLOW}Running in TEST MODE (10 stocks with parallel download)...${NC}"
        python3 data_fetcher.py --test --db-path "$DB_PATH" --workers 5
    else
        echo -e "${YELLOW}Downloading full ASX200 data with parallel download (5-10 minutes)...${NC}"
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
echo -e "${BLUE}Step 2/5: Running v1 (Current) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 backtest_v1.py \
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
V1_METRICS=$(ls -t results/backtest_v1_metrics_*.json | head -1)
V1_TRADES=$(ls -t results/backtest_v1_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $V1_METRICS${NC}"
echo -e "${GREEN}  Trades:  $V1_TRADES${NC}"
echo ""

################################################################################
# Step 3: Run v2 Strategy Backtest
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 3/5: Running v2 (Enhanced) Strategy Backtest${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 backtest_v2.py \
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
V2_METRICS=$(ls -t results/backtest_v2_metrics_*.json | head -1)
V2_TRADES=$(ls -t results/backtest_v2_trades_*.csv | head -1)

echo -e "${GREEN}  Metrics: $V2_METRICS${NC}"
echo -e "${GREEN}  Trades:  $V2_TRADES${NC}"
echo ""

################################################################################
# Step 4: Generate Comparison Reports
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 4/5: Generating Comparison Reports${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 compare_strategies.py \
    --v1-metrics "$V1_METRICS" \
    --v1-trades "$V1_TRADES" \
    --v2-metrics "$V2_METRICS" \
    --v2-trades "$V2_TRADES" \
    --output-dir ./comparison_reports \
    --format all

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Comparison reports generated successfully${NC}"
else
    echo -e "${RED}✗ Comparison report generation failed${NC}"
    exit 1
fi

# Find the most recent comparison reports
COMPARISON_HTML=$(ls -t comparison_reports/comparison_report_*.html | head -1)
COMPARISON_TXT=$(ls -t comparison_reports/comparison_summary_*.txt | head -1)
COMPARISON_CSV=$(ls -t comparison_reports/comparison_metrics_*.csv | head -1)

echo -e "${GREEN}  HTML:    $COMPARISON_HTML${NC}"
echo -e "${GREEN}  Summary: $COMPARISON_TXT${NC}"
echo -e "${GREEN}  CSV:     $COMPARISON_CSV${NC}"
echo ""

################################################################################
# Step 5: Display Summary
################################################################################

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 5/5: Executive Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Display the text summary
if [ -f "$COMPARISON_TXT" ]; then
    cat "$COMPARISON_TXT"
else
    echo -e "${YELLOW}Summary file not found${NC}"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}✓ Complete Workflow Finished Successfully!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Display file locations
echo -e "${GREEN}Results Available At:${NC}"
echo ""
echo -e "${YELLOW}v1 Backtest Results:${NC}"
echo -e "  • Trades:  $V1_TRADES"
echo -e "  • Metrics: $V1_METRICS"
echo -e "  • Report:  ${V1_METRICS%.json}.txt"
echo ""
echo -e "${YELLOW}v2 Backtest Results:${NC}"
echo -e "  • Trades:  $V2_TRADES"
echo -e "  • Metrics: $V2_METRICS"
echo -e "  • Report:  ${V2_METRICS%.json}.txt"
echo ""
echo -e "${YELLOW}Comparison Reports:${NC}"
echo -e "  • HTML:    $COMPARISON_HTML"
echo -e "  • Summary: $COMPARISON_TXT"
echo -e "  • CSV:     $COMPARISON_CSV"
echo ""

# Open HTML report (if available)
if command -v open &> /dev/null && [ -f "$COMPARISON_HTML" ]; then
    echo -e "${GREEN}Opening HTML report in browser...${NC}"
    open "$COMPARISON_HTML"
elif command -v xdg-open &> /dev/null && [ -f "$COMPARISON_HTML" ]; then
    echo -e "${GREEN}Opening HTML report in browser...${NC}"
    xdg-open "$COMPARISON_HTML"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Thank you for using the ASX200 Backtesting Framework!${NC}"
echo -e "${BLUE}================================================================${NC}"
