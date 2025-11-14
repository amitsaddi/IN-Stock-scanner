#!/bin/bash

################################################################################
# IN-testing Framework Verification Script
# Verifies that the core infrastructure is properly set up
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   IN-testing Framework - Setup Verification${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        ((ERRORS++))
    fi
}

# Function to check if a directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
    else
        echo -e "${RED}✗${NC} $1/ (MISSING)"
        ((ERRORS++))
    fi
}

# Function to check if a file is executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (executable)"
    else
        echo -e "${YELLOW}⚠${NC} $1 (not executable)"
        ((WARNINGS++))
    fi
}

echo -e "${BLUE}Checking folder structure...${NC}"
check_dir "scripts"
check_dir "backtesting"
check_dir "backtesting/results"
check_dir "backtesting/comparison_reports"
check_dir "data"
check_dir "logs"
echo ""

echo -e "${BLUE}Checking core infrastructure files...${NC}"
check_file "scripts/db_schema.py"
check_file "scripts/data_fetcher.py"
check_file "backtesting/run_full_backtest.sh"
check_file "requirements.txt"
echo ""

echo -e "${BLUE}Checking documentation files...${NC}"
check_file "START_HERE.md"
check_file "IMPLEMENTATION_GUIDE.md"
check_file "PROJECT_SUMMARY.md"
echo ""

echo -e "${BLUE}Checking script permissions...${NC}"
check_executable "backtesting/run_full_backtest.sh"
echo ""

echo -e "${BLUE}Checking Python syntax...${NC}"
if command -v python3 &> /dev/null; then
    if python3 -m py_compile scripts/db_schema.py 2>/dev/null; then
        echo -e "${GREEN}✓${NC} db_schema.py (valid Python)"
    else
        echo -e "${RED}✗${NC} db_schema.py (syntax error)"
        ((ERRORS++))
    fi
    
    if python3 -m py_compile scripts/data_fetcher.py 2>/dev/null; then
        echo -e "${GREEN}✓${NC} data_fetcher.py (valid Python)"
    else
        echo -e "${RED}✗${NC} data_fetcher.py (syntax error)"
        ((ERRORS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} python3 not found (skipping syntax check)"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}Checking virtual environment...${NC}"
VENV_PATH="../stockScannerIN"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found at $VENV_PATH"
elif [ -f "$VENV_PATH/Scripts/activate" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found at $VENV_PATH (Windows)"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not found at $VENV_PATH"
    echo -e "${YELLOW}  ${NC} Script will use system Python"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}Checking Python packages (if available)...${NC}"
if command -v python3 &> /dev/null; then
    MISSING_PACKAGES=()
    
    for pkg in pandas numpy yfinance scipy matplotlib plotly; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $pkg installed"
        else
            echo -e "${YELLOW}⚠${NC} $pkg not installed"
            MISSING_PACKAGES+=("$pkg")
            ((WARNINGS++))
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}To install missing packages:${NC}"
        echo -e "  pip install ${MISSING_PACKAGES[@]}"
    fi
else
    echo -e "${YELLOW}⚠${NC} python3 not found (skipping package check)"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Verification Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ PERFECT!${NC} All checks passed."
    echo ""
    echo -e "${GREEN}Core infrastructure is ready.${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "1. Read START_HERE.md for overview"
    echo -e "2. Read IMPLEMENTATION_GUIDE.md for next steps"
    echo -e "3. Implement the 6 backtest Python scripts"
    echo -e "4. Test with: cd backtesting && ./run_full_backtest.sh --test"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ GOOD${NC} with $WARNINGS warning(s)."
    echo ""
    echo -e "${YELLOW}Core infrastructure is functional but has minor issues.${NC}"
    echo ""
    echo -e "You can proceed, but address warnings for best experience."
else
    echo -e "${RED}✗ ISSUES FOUND${NC}: $ERRORS error(s), $WARNINGS warning(s)."
    echo ""
    echo -e "${RED}Please fix errors before proceeding.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Core Infrastructure Status: COMPLETE (40%)${NC}"
echo -e "${YELLOW}Remaining Work: 6 backtest/comparison scripts (60%)${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

