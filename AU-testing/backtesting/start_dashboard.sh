#!/bin/bash

# Start the Backtest Results Dashboard
# This script activates the virtual environment and launches the Flask dashboard

echo "Starting ASX200 Backtest Results Dashboard..."
echo "=============================================="

# Path to virtual environment
VENV_PATH="../../stockScannerIN"

# Activate virtual environment if it exists
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Using system Python..."
fi

# Default database path
DB_PATH="../data/asx200_historical.db"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    echo "Please run the data fetcher first to create the database."
    exit 1
fi

# Launch dashboard
echo ""
echo "Launching dashboard..."
echo "Database: $DB_PATH"
echo "Port: 5000"
echo ""
echo "Dashboard will be available at: http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the dashboard"
echo ""

python3 view_results_dashboard.py --db-path "$DB_PATH" --port 5000
