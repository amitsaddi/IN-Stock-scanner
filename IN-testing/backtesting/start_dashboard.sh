#!/bin/bash
# Start the Backtest Results Dashboard
# Usage: ./start_dashboard.sh

echo "========================================="
echo "Backtest Results Dashboard"
echo "========================================="

# Activate virtual environment
VENV_PATH="../../stockScannerIN"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Continuing without virtual environment..."
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask not installed. Installing requirements..."
    pip install -r ../requirements.txt
fi

# Start dashboard
echo "Starting dashboard on http://127.0.0.1:5001"
echo "Press Ctrl+C to stop"
echo ""

python3 view_results_dashboard.py --db-path ../data/nifty500_historical.db --port 5001
