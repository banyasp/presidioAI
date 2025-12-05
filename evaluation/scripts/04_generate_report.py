#!/usr/bin/env python3
"""
Script to generate evaluation report from database results.

Usage:
    python 04_generate_report.py --db-path scotus_cases.db --run-id <run_id> --output-dir evaluation_results/reports
"""

import argparse
import sys
import os
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.visualization.reports import generate_text_report, generate_comparison_table
from evaluation.storage.database import get_db_connection


def load_results_from_db(db_path: str, run_id: str) -> dict:
    """Load evaluation results from database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get run info
    cursor.execute("""
        SELECT DISTINCT model_name, run_timestamp
        FROM evaluation_results
        WHERE run_id = ?
    """, (run_id,))

    rows = cursor.fetchall()
    if not rows:
        raise ValueError(f"No results found for run_id: {run_id}")

    timestamp = rows[0]['run_timestamp']

    # Build results dictionary
    results = {
        'run_id': run_id,
        'timestamp': timestamp,
        'models': {}
    }

    for row in rows:
        model_name = row['model_name']

        # Get aggregate metrics
        cursor.execute("""
            SELECT metric_name, metric_value
            FROM evaluation_results
            WHERE run_id = ? AND model_name = ?
        """, (run_id, model_name))

        metrics = {r['metric_name']: r['metric_value'] for r in cursor.fetchall()}

        # Count queries
        cursor.execute("""
            SELECT COUNT(DISTINCT query_id) as count
            FROM evaluation_query_metrics
            WHERE run_id = ? AND model_name = ?
        """, (run_id, model_name))

        num_queries = cursor.fetchone()['count']

        results['models'][model_name] = {
            'aggregate_metrics': metrics,
            'num_queries': num_queries
        }

    conn.close()
    return results


def list_available_runs(db_path: str):
    """List all available evaluation runs."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT run_id, run_timestamp, COUNT(DISTINCT model_name) as num_models
        FROM evaluation_results
        GROUP BY run_id
        ORDER BY run_timestamp DESC
    """)

    runs = cursor.fetchall()
    conn.close()

    if not runs:
        print("No evaluation runs found in database.")
        return

    print("\nAvailable evaluation runs:")
    print("-" * 80)
    for run in runs:
        run_id_short = run['run_id'][:8]
        print(f"  {run_id_short}... | {run['run_timestamp']} | {run['num_models']} models")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--run-id", help="Run ID to generate report for (use --list to see available)")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--latest", action="store_true", help="Use latest run")
    parser.add_argument("--output-dir", default="evaluation_results/reports", help="Output directory")

    args = parser.parse_args()

    # Validate database exists
    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found: {args.db_path}")
        sys.exit(1)

    # List runs if requested
    if args.list:
        list_available_runs(args.db_path)
        sys.exit(0)

    # Get run ID
    run_id = args.run_id

    if args.latest:
        # Get latest run
        conn = get_db_connection(args.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT run_id
            FROM evaluation_results
            ORDER BY run_timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("ERROR: No evaluation runs found in database")
            sys.exit(1)

        run_id = row['run_id']
        print(f"Using latest run: {run_id[:8]}...")

    if not run_id:
        print("ERROR: Must specify --run-id or --latest")
        print("Use --list to see available runs")
        sys.exit(1)

    # Load results
    print(f"Loading results for run: {run_id[:8]}...")
    try:
        results = load_results_from_db(args.db_path, run_id)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate report
    report_path = os.path.join(args.output_dir, f"report_{run_id[:8]}.txt")
    print(f"\nGenerating report...")
    generate_text_report(results, report_path)

    print(f"\n✓ Report generation complete!")


if __name__ == "__main__":
    main()
