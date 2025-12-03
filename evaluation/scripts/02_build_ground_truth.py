#!/usr/bin/env python3
"""
Script to build ground truth from case citations.

Usage:
    python 02_build_ground_truth.py --db-path scotus_cases.db [--source-relevance 2] [--citation-relevance 1]
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.ground_truth.builder import populate_ground_truth_table, print_sample_ground_truth
from evaluation.storage.database import get_ground_truth_count, get_query_count


def main():
    parser = argparse.ArgumentParser(description="Build ground truth from case citations")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--source-relevance", type=int, default=2, help="Relevance score for source case")
    parser.add_argument("--citation-relevance", type=int, default=1, help="Relevance score for cited cases")
    parser.add_argument("--match-threshold", type=float, default=0.85, help="Fuzzy matching threshold (0-1)")
    parser.add_argument("--show-samples", action="store_true", help="Show sample ground truth after building")

    args = parser.parse_args()

    # Validate database exists
    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found: {args.db_path}")
        sys.exit(1)

    # Check if we have queries
    query_count = get_query_count(args.db_path)
    if query_count == 0:
        print("ERROR: No queries found in database. Run 01_generate_queries.py first.")
        sys.exit(1)

    # Check existing ground truth
    existing_count = get_ground_truth_count(args.db_path)
    if existing_count > 0:
        response = input(f"WARNING: Found {existing_count} existing ground truth entries. Clear and rebuild? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

        # Clear existing ground truth
        import sqlite3
        conn = sqlite3.connect(args.db_path)
        conn.execute("DELETE FROM evaluation_ground_truth")
        conn.commit()
        conn.close()
        print("✓ Cleared existing ground truth")

    print("="*80)
    print("GROUND TRUTH CONSTRUCTION")
    print("="*80)
    print(f"Database: {args.db_path}")
    print(f"Queries: {query_count}")
    print(f"Source relevance: {args.source_relevance}")
    print(f"Citation relevance: {args.citation_relevance}")
    print(f"Match threshold: {args.match_threshold}")
    print("="*80)
    print()

    # Build ground truth
    stats = populate_ground_truth_table(
        args.db_path,
        source_relevance=args.source_relevance,
        citation_relevance=args.citation_relevance,
        match_threshold=args.match_threshold
    )

    print()
    print("="*80)
    print("STATISTICS")
    print("="*80)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)

    # Optionally show samples
    if args.show_samples:
        print_sample_ground_truth(args.db_path, n=5)


if __name__ == "__main__":
    main()
