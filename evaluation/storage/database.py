"""Database schema and operations for evaluation framework."""

import sqlite3
import os
from typing import Optional


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Create a database connection with Row factory.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_evaluation_schema(db_path: str) -> None:
    """
    Create evaluation tables in the database if they don't exist.

    Tables created:
    - evaluation_queries: Synthetic queries generated from cases
    - evaluation_ground_truth: Relevance scores for query-case pairs
    - evaluation_results: Aggregate metrics per model
    - evaluation_query_metrics: Per-query metrics for error analysis

    Args:
        db_path: Path to SQLite database file
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Table 1: Evaluation queries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_case_id INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            query_type TEXT NOT NULL,
            generation_method TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_case_id) REFERENCES cases(id)
        )
    """)

    # Table 2: Ground truth (relevance scores)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_ground_truth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            case_id INTEGER NOT NULL,
            relevance_score INTEGER NOT NULL,
            FOREIGN KEY (query_id) REFERENCES evaluation_queries(id),
            FOREIGN KEY (case_id) REFERENCES cases(id),
            UNIQUE(query_id, case_id)
        )
    """)

    # Table 3: Aggregate evaluation results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 4: Per-query metrics (for error analysis)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_query_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            query_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            FOREIGN KEY (query_id) REFERENCES evaluation_queries(id)
        )
    """)

    conn.commit()

    # Create indices for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ground_truth_query
        ON evaluation_ground_truth(query_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ground_truth_case
        ON evaluation_ground_truth(case_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_run
        ON evaluation_results(run_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_metrics_run
        ON evaluation_query_metrics(run_id, query_id)
    """)

    conn.commit()
    conn.close()

    print(f"✓ Evaluation schema initialized in {db_path}")
    print("  - evaluation_queries")
    print("  - evaluation_ground_truth")
    print("  - evaluation_results")
    print("  - evaluation_query_metrics")


def clear_evaluation_data(db_path: str, confirm: bool = False) -> None:
    """
    Clear all evaluation data from the database.

    WARNING: This will delete all queries, ground truth, and results!

    Args:
        db_path: Path to SQLite database file
        confirm: Must be True to actually delete data
    """
    if not confirm:
        print("ERROR: Must pass confirm=True to clear evaluation data")
        return

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM evaluation_query_metrics")
    cursor.execute("DELETE FROM evaluation_results")
    cursor.execute("DELETE FROM evaluation_ground_truth")
    cursor.execute("DELETE FROM evaluation_queries")

    conn.commit()
    conn.close()

    print("✓ Cleared all evaluation data")


def get_query_count(db_path: str) -> int:
    """Get the number of queries in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM evaluation_queries")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_ground_truth_count(db_path: str) -> int:
    """Get the number of ground truth entries in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM evaluation_ground_truth")
    count = cursor.fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    # Test initialization
    import sys

    if len(sys.argv) < 2:
        print("Usage: python database.py <path_to_db>")
        sys.exit(1)

    db_path = sys.argv[1]
    initialize_evaluation_schema(db_path)

    # Print status
    print(f"\nStatus:")
    print(f"  Queries: {get_query_count(db_path)}")
    print(f"  Ground truth entries: {get_ground_truth_count(db_path)}")
