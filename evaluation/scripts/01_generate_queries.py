#!/usr/bin/env python3
"""
Script to generate synthetic queries from legal cases.

Usage:
    python 01_generate_queries.py --db-path scotus_cases.db [--model llama3.2:1b] [--extractive-only] [--generative-only]
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.query_generation.generator import generate_queries_for_all_cases, print_sample_queries
from evaluation.storage.database import get_query_count


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic queries from legal cases")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name")
    parser.add_argument("--extractive-only", action="store_true", help="Generate only extractive queries")
    parser.add_argument("--generative-only", action="store_true", help="Generate only generative queries (requires Ollama)")
    parser.add_argument("--show-samples", action="store_true", help="Show sample queries after generation")

    args = parser.parse_args()

    # Validate database exists
    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found: {args.db_path}")
        sys.exit(1)

    # Check existing queries
    existing_count = get_query_count(args.db_path)
    if existing_count > 0:
        response = input(f"WARNING: Found {existing_count} existing queries. Continue and add more? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Determine which types to generate
    use_extractive = not args.generative_only
    use_generative = not args.extractive_only

    if args.extractive_only and args.generative_only:
        print("ERROR: Cannot use both --extractive-only and --generative-only")
        sys.exit(1)

    print("="*80)
    print("QUERY GENERATION")
    print("="*80)
    print(f"Database: {args.db_path}")
    print(f"Extractive queries: {use_extractive}")
    print(f"Generative queries: {use_generative}")
    if use_generative:
        print(f"Model: {args.model}")
    print("="*80)
    print()

    # Generate queries
    count = generate_queries_for_all_cases(
        args.db_path,
        model=args.model,
        use_extractive=use_extractive,
        use_generative=use_generative
    )

    print()
    print("="*80)
    print(f"✓ Successfully generated {count} queries")
    print(f"✓ Total queries in database: {get_query_count(args.db_path)}")
    print("="*80)

    # Optionally show samples
    if args.show_samples:
        print_sample_queries(args.db_path)


if __name__ == "__main__":
    main()
