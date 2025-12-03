#!/usr/bin/env python3
"""
All-in-one script to run the complete evaluation pipeline.

Usage:
    python run_full_evaluation.py --db-path scotus_cases.db [--skip-query-generation] [--skip-ground-truth]
"""

import argparse
import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation pipeline")
    parser.add_argument("--db-path", default="scotus_cases.db", help="Path to database")
    parser.add_argument("--skip-query-generation", action="store_true", help="Skip query generation step")
    parser.add_argument("--skip-ground-truth", action="store_true", help="Skip ground truth building step")
    parser.add_argument("--extractive-only", action="store_true", help="Generate only extractive queries")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (default: all)")
    parser.add_argument("--cache-dir", default="evaluation_results/cache", help="Cache directory")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = args.db_path

    print("="*80)
    print("COMPLETE EVALUATION PIPELINE")
    print("="*80)
    print(f"Database: {db_path}")
    print(f"Skip query generation: {args.skip_query_generation}")
    print(f"Skip ground truth: {args.skip_ground_truth}")
    print(f"Extractive only: {args.extractive_only}")
    print("="*80)

    # Step 1: Query Generation
    if not args.skip_query_generation:
        cmd = f"python {script_dir}/01_generate_queries.py --db-path {db_path}"
        if args.extractive_only:
            cmd += " --extractive-only"
        cmd += " --show-samples"

        if not run_command(cmd, "STEP 1: Generating Queries"):
            return

    # Step 2: Ground Truth Building
    if not args.skip_ground_truth:
        cmd = f"python {script_dir}/02_build_ground_truth.py --db-path {db_path} --show-samples"

        if not run_command(cmd, "STEP 2: Building Ground Truth"):
            return

    # Step 3: Run Evaluation
    cmd = f"python {script_dir}/03_run_evaluation.py --db-path {db_path} --cache-dir {args.cache_dir}"
    if args.models:
        cmd += f" --models {' '.join(args.models)}"

    if not run_command(cmd, "STEP 3: Running Evaluation"):
        return

    # Step 4: Generate Report
    cmd = f"python {script_dir}/04_generate_report.py --db-path {db_path} --latest --output-dir evaluation_results/reports"

    if not run_command(cmd, "STEP 4: Generating Report"):
        return

    print("\n" + "="*80)
    print("✓ COMPLETE EVALUATION PIPELINE FINISHED SUCCESSFULLY")
    print("="*80)
    print("\nResults:")
    print(f"  - Cached predictions: {args.cache_dir}/")
    print(f"  - Evaluation results: {db_path} (evaluation_results table)")
    print(f"  - Report: evaluation_results/reports/")
    print("="*80)


if __name__ == "__main__":
    main()
