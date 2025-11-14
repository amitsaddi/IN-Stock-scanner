"""
Web Dashboard for Viewing Backtest Results History
Displays trends and comparisons for both Swing and BTST strategies

Usage:
    python view_results_dashboard.py --db-path ../data/nifty500_historical.db --port 5001
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Global database path
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


@app.route('/api/overview')
def get_overview():
    """Get overview stats for both strategies"""
    conn = get_db_connection()
    cursor = conn.cursor()

    overview = {}

    for strategy_type in ['swing', 'btst']:
        # Get latest comparison
        cursor.execute("""
            SELECT * FROM comparison_results_history
            WHERE strategy_type = ?
            ORDER BY run_date DESC
            LIMIT 1
        """, (strategy_type,))

        latest = cursor.fetchone()

        if latest:
            overview[strategy_type] = {
                'run_date': latest['run_date'],
                'recommendation': latest['recommendation'],
                'confidence_level': latest['confidence_level'],
                'criteria_met': latest['criteria_met'],
                'total_criteria': latest['total_criteria'],
                'delta_avg_return': round(latest['delta_avg_return'], 2),
                'delta_sharpe_ratio': round(latest['delta_sharpe_ratio'], 2),
                'delta_win_rate': round(latest['delta_win_rate'], 2)
            }
        else:
            overview[strategy_type] = None

    conn.close()
    return jsonify(overview)


@app.route('/api/trend/<strategy_type>')
def get_trend(strategy_type):
    """Get 30-day trend for a strategy"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            run_date,
            recommendation,
            confidence_level,
            criteria_met,
            total_criteria,
            delta_avg_return,
            delta_sharpe_ratio
        FROM comparison_results_history
        WHERE strategy_type = ?
        AND run_date >= date('now', '-30 days')
        ORDER BY run_date ASC
    """, (strategy_type,))

    rows = cursor.fetchall()
    trend = []

    for row in rows:
        trend.append({
            'date': row['run_date'],
            'recommendation': row['recommendation'],
            'confidence': row['confidence_level'],
            'criteria_met': row['criteria_met'],
            'total_criteria': row['total_criteria'],
            'delta_avg_return': round(row['delta_avg_return'], 2),
            'delta_sharpe_ratio': round(row['delta_sharpe_ratio'], 2)
        })

    conn.close()
    return jsonify(trend)


@app.route('/api/performance/<strategy_type>/<version>')
def get_performance(strategy_type, version):
    """Get latest performance metrics for a specific version"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM backtest_results_history
        WHERE strategy_type = ? AND version = ?
        ORDER BY run_date DESC
        LIMIT 1
    """, (strategy_type, version))

    row = cursor.fetchone()

    if row:
        performance = {
            'run_date': row['run_date'],
            'total_trades': row['total_trades'],
            'winning_trades': row['winning_trades'],
            'losing_trades': row['losing_trades'],
            'win_rate': round(row['win_rate'], 2),
            'avg_return': round(row['avg_return'], 2),
            'total_return': round(row['total_return'], 2),
            'sharpe_ratio': round(row['sharpe_ratio'], 2),
            'max_drawdown': round(row['max_drawdown'], 2),
            'avg_hold_days': round(row['avg_hold_days'], 1)
        }
    else:
        performance = None

    conn.close()
    return jsonify(performance)


@app.route('/api/history/<strategy_type>')
def get_history(strategy_type):
    """Get full comparison history for a strategy"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            run_date,
            recommendation,
            confidence_level,
            criteria_met,
            total_criteria,
            delta_total_trades,
            delta_win_rate,
            delta_avg_return,
            delta_sharpe_ratio,
            delta_max_drawdown,
            returns_ttest_pvalue,
            winrate_chisquare_pvalue
        FROM comparison_results_history
        WHERE strategy_type = ?
        ORDER BY run_date DESC
        LIMIT 30
    """, (strategy_type,))

    rows = cursor.fetchall()
    history = []

    for row in rows:
        history.append({
            'date': row['run_date'],
            'recommendation': row['recommendation'],
            'confidence': row['confidence_level'],
            'criteria_met': f"{row['criteria_met']}/{row['total_criteria']}",
            'delta_trades': row['delta_total_trades'],
            'delta_win_rate': round(row['delta_win_rate'], 2),
            'delta_avg_return': round(row['delta_avg_return'], 2),
            'delta_sharpe': round(row['delta_sharpe_ratio'], 2),
            'delta_drawdown': round(row['delta_max_drawdown'], 2),
            'returns_pvalue': round(row['returns_ttest_pvalue'], 4),
            'winrate_pvalue': round(row['winrate_chisquare_pvalue'], 4)
        })

    conn.close()
    return jsonify(history)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='View Backtest Results Dashboard')
    parser.add_argument('--db-path', type=str,
                       default='../data/nifty500_historical.db',
                       help='Path to database file')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run dashboard on (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')

    args = parser.parse_args()

    # Set global database path
    global DB_PATH
    db_file = Path(args.db_path)
    if not db_file.is_absolute():
        db_file = Path(__file__).parent / args.db_path
    DB_PATH = str(db_file.resolve())

    print("=" * 80)
    print("Backtest Results Dashboard")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the server")
    print()

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
