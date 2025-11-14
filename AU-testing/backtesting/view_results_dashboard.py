"""
Web Dashboard for Viewing Backtest Results History
Displays trends, comparisons, and recommendations from the results tracking database
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global config
DB_PATH = None


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/summary')
def api_summary():
    """Get summary statistics for the dashboard"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get latest comparison
    cursor.execute("""
        SELECT * FROM comparison_results_history
        ORDER BY run_date DESC
        LIMIT 1
    """)
    latest_comparison = cursor.fetchone()

    # Get 30-day trend
    cursor.execute("""
        SELECT * FROM vw_30day_trend
        ORDER BY run_date DESC
    """)
    trend_data = cursor.fetchall()

    # Get total runs
    cursor.execute("SELECT COUNT(*) as count FROM comparison_results_history")
    total_runs = cursor.fetchone()['count']

    # Get v1 vs v2 performance over time
    cursor.execute("""
        SELECT run_date, version, avg_return, win_rate, sharpe_ratio
        FROM backtest_results_history
        ORDER BY run_date DESC
        LIMIT 60
    """)
    performance_history = cursor.fetchall()

    conn.close()

    # Format data
    summary = {
        'latest_comparison': dict(latest_comparison) if latest_comparison else None,
        'total_runs': total_runs,
        'trend_30day': [dict(row) for row in trend_data],
        'performance_history': [dict(row) for row in performance_history]
    }

    return jsonify(summary)


@app.route('/api/results/<run_date>')
def api_results(run_date):
    """Get detailed results for a specific run date"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get comparison results
    cursor.execute("""
        SELECT * FROM comparison_results_history
        WHERE run_date = ?
    """, (run_date,))
    comparison = cursor.fetchone()

    # Get v1 and v2 results
    cursor.execute("""
        SELECT * FROM backtest_results_history
        WHERE run_date = ?
        ORDER BY version
    """, (run_date,))
    results = cursor.fetchall()

    conn.close()

    data = {
        'comparison': dict(comparison) if comparison else None,
        'results': [dict(row) for row in results]
    }

    return jsonify(data)


@app.route('/api/dates')
def api_dates():
    """Get list of all run dates"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT run_date
        FROM comparison_results_history
        ORDER BY run_date DESC
    """)
    dates = cursor.fetchall()

    conn.close()

    return jsonify([row['run_date'] for row in dates])


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='View Backtest Results Dashboard')
    parser.add_argument('--db-path', type=str, required=True,
                       help='Path to SQLite database with results history')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run dashboard on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run dashboard on (default: 127.0.0.1)')

    args = parser.parse_args()

    global DB_PATH
    DB_PATH = args.db_path

    logger.info(f"Starting dashboard with database: {DB_PATH}")
    logger.info(f"Dashboard will be available at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
