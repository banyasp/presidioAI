#!/usr/bin/env python3
"""
Script to run evaluation on all models.

Usage:
    python 03_run_evaluation.py --db-path scotus_cases.db [--cache-dir evaluation_results/cache] [--models legal-bert harvard-bert]
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.pipeline.runner import EvaluationPipeline
from evaluation.storage.database import get_query_count, get_ground_truth_count


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--cache-dir", default="evaluation_results/cache", help="Cache directory")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (default: all 3 models)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1,3,5,10,20], help="K values for metrics")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (slower)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")

    args = parser.parse_args()

    # Validate database exists
    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found: {args.db_path}")
        sys.exit(1)

    # Check prerequisites
    query_count = get_query_count(args.db_path)
    if query_count == 0:
        print("ERROR: No queries found. Run 01_generate_queries.py first.")
        sys.exit(1)

    gt_count = get_ground_truth_count(args.db_path)
    if gt_count == 0:
        print("ERROR: No ground truth found. Run 02_build_ground_truth.py first.")
        sys.exit(1)

    print("="*80)
    print("EVALUATION PIPELINE")
    print("="*80)
    print(f"Database: {args.db_path}")
    print(f"Queries: {query_count}")
    print(f"Ground truth entries: {gt_count}")
    print(f"Models: {args.models or 'all (legal-bert, harvard-bert, sentence-transformer)'}")
    print(f"K values: {args.k_values}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("="*80)
    print()

    # Initialize pipeline
    pipeline = EvaluationPipeline(args.db_path, args.cache_dir)

    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...")
        pipeline.cache_manager.clear()

    # Run evaluation
    results = pipeline.run_full_evaluation(
        models=args.models,
        k_values=args.k_values,
        use_cache=not args.no_cache,
        save_to_db=True
    )

    # Print comparison
    pipeline.print_comparison(results)

    # Print cache stats
    cache_stats = pipeline.cache_manager.get_cache_stats()
    print("\nCache statistics:")
    for model, count in cache_stats.items():
        print(f"  {model}: {count} cached queries")

    print(f"\n✓ Evaluation complete! Run ID: {results['run_id']}")
    print(f"  Results saved to database: {args.db_path}")


if __name__ == "__main__":
    main()
