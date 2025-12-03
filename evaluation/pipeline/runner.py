"""Main evaluation pipeline for running and comparing models."""

import sys
import os
import sqlite3
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import uuid
from datetime import datetime

# Add parent directories to path to import backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.logic import find_similar_cases, load_model_resources
from evaluation.pipeline.cache import CacheManager
from evaluation.metrics.compute import MetricsAggregator
from evaluation.storage.database import get_db_connection
from evaluation.ground_truth.builder import get_ground_truth_for_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main pipeline for evaluating similarity models."""

    def __init__(self, db_path: str, cache_dir: str = "evaluation_results/cache"):
        """
        Initialize evaluation pipeline.

        Args:
            db_path: Path to SQLite database
            cache_dir: Directory for caching results
        """
        self.db_path = db_path
        self.cache_manager = CacheManager(cache_dir)
        self.models = ['legal-bert', 'harvard-bert', 'sentence-transformer']

    def load_queries(self) -> List[Dict]:
        """
        Load all queries from database.

        Returns:
            List of query dictionaries
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, query_text, source_case_id
            FROM evaluation_queries
            ORDER BY id
        """)

        queries = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return queries

    def get_case_id_by_name(self, case_name: str) -> Optional[int]:
        """Map case name to case ID."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM cases WHERE case_name = ?", (case_name,))
        row = cursor.fetchone()
        conn.close()

        return row['id'] if row else None

    def run_retrieval(self, query_text: str, model_name: str, top_k: int = 100) -> List[int]:
        """
        Run retrieval using backend logic.

        Args:
            query_text: Query string
            model_name: Model to use
            top_k: Number of results to return

        Returns:
            List of case IDs ranked by similarity
        """
        # Use existing backend function
        results = find_similar_cases(query_text, model_name, top_k=top_k)

        # Extract case IDs from results
        case_ids = []
        for result in results:
            case_name = result['case_name']
            case_id = self.get_case_id_by_name(case_name)
            if case_id:
                case_ids.append(case_id)

        return case_ids

    def evaluate_model(
        self,
        model_name: str,
        k_values: List[int] = [1, 3, 5, 10, 20],
        use_cache: bool = True,
        top_k_retrieval: int = 100
    ) -> Dict:
        """
        Evaluate a single model on all queries.

        Args:
            model_name: Model to evaluate
            k_values: List of k values for metrics
            use_cache: Whether to use cached results
            top_k_retrieval: How many results to retrieve per query

        Returns:
            Dictionary with aggregate metrics and per-query DataFrame
        """
        logger.info(f"Evaluating model: {model_name}")

        # Preload model resources
        logger.info("Preloading model resources...")
        load_model_resources(model_name)

        # Load queries
        queries = self.load_queries()
        logger.info(f"Loaded {len(queries)} queries")

        # Initialize metrics aggregator
        aggregator = MetricsAggregator(k_values=k_values)

        # Evaluate each query
        for query in tqdm(queries, desc=f"Evaluating {model_name}"):
            query_id = query['id']
            query_text = query['query_text']

            # Check cache
            if use_cache:
                cached_results = self.cache_manager.get(model_name, query_id)
                if cached_results is not None:
                    results = cached_results
                else:
                    results = self.run_retrieval(query_text, model_name, top_k_retrieval)
                    self.cache_manager.save(model_name, query_id, results)
            else:
                results = self.run_retrieval(query_text, model_name, top_k_retrieval)

            # Get ground truth
            ground_truth = get_ground_truth_for_query(self.db_path, query_id)

            # Compute metrics
            aggregator.add_query_result(query_id, results, ground_truth)

        # Get results
        aggregate_metrics = aggregator.get_aggregate_metrics()
        per_query_df = aggregator.get_per_query_dataframe()

        logger.info(f"✓ Completed evaluation for {model_name}")

        return {
            'model_name': model_name,
            'aggregate_metrics': aggregate_metrics,
            'per_query_metrics': per_query_df,
            'num_queries': len(queries)
        }

    def run_full_evaluation(
        self,
        models: Optional[List[str]] = None,
        k_values: List[int] = [1, 3, 5, 10, 20],
        use_cache: bool = True,
        save_to_db: bool = True
    ) -> Dict[str, Dict]:
        """
        Run evaluation on all models.

        Args:
            models: List of model names (default: all 3 models)
            k_values: List of k values for metrics
            use_cache: Whether to use cached results
            save_to_db: Whether to save results to database

        Returns:
            Dictionary mapping model_name -> results
        """
        if models is None:
            models = self.models

        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        logger.info("="*80)
        logger.info("STARTING FULL EVALUATION")
        logger.info("="*80)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Models: {', '.join(models)}")
        logger.info(f"K values: {k_values}")
        logger.info(f"Use cache: {use_cache}")
        logger.info("="*80)

        results = {}

        for model_name in models:
            model_results = self.evaluate_model(
                model_name,
                k_values=k_values,
                use_cache=use_cache
            )
            results[model_name] = model_results

            # Optionally save to database
            if save_to_db:
                self._save_results_to_db(
                    run_id,
                    timestamp,
                    model_name,
                    model_results['aggregate_metrics'],
                    model_results['per_query_metrics']
                )

        logger.info("="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)

        return {
            'run_id': run_id,
            'timestamp': timestamp,
            'models': results
        }

    def _save_results_to_db(
        self,
        run_id: str,
        timestamp: str,
        model_name: str,
        aggregate_metrics: Dict[str, float],
        per_query_metrics
    ):
        """Save evaluation results to database."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        # Save aggregate metrics
        for metric_name, metric_value in aggregate_metrics.items():
            cursor.execute("""
                INSERT INTO evaluation_results (run_id, model_name, metric_name, metric_value, run_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, model_name, metric_name, float(metric_value), timestamp))

        # Save per-query metrics
        for _, row in per_query_metrics.iterrows():
            query_id = int(row['query_id'])
            for metric_name in per_query_metrics.columns:
                if metric_name == 'query_id':
                    continue
                metric_value = float(row[metric_name])
                cursor.execute("""
                    INSERT INTO evaluation_query_metrics (run_id, query_id, model_name, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?)
                """, (run_id, query_id, model_name, metric_name, metric_value))

        conn.commit()
        conn.close()

        logger.info(f"✓ Saved results to database (run_id: {run_id[:8]}...)")

    def print_comparison(self, results: Dict[str, Dict]):
        """
        Print a comparison table of all models.

        Args:
            results: Results from run_full_evaluation()
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print()

        # Extract k values from first model
        first_model = list(results['models'].values())[0]
        k_values = [1, 3, 5, 10, 20]  # Standard k values

        # Print header
        print(f"{'Model':<25} {'P@1':>8} {'P@3':>8} {'P@5':>8} {'MRR':>8} {'NDCG@3':>8} {'NDCG@10':>8}")
        print("-" * 80)

        # Print each model
        for model_name, model_results in results['models'].items():
            metrics = model_results['aggregate_metrics']

            row = f"{model_name:<25}"
            row += f" {metrics.get('precision@1', 0):.4f}"
            row += f" {metrics.get('precision@3', 0):.4f}"
            row += f" {metrics.get('precision@5', 0):.4f}"
            row += f" {metrics.get('mrr', 0):.4f}"
            row += f" {metrics.get('ndcg@3', 0):.4f}"
            row += f" {metrics.get('ndcg@10', 0):.4f}"

            print(row)

        print("="*80)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--db-path", default="scotus_cases.db", help="Path to database")
    parser.add_argument("--cache-dir", default="evaluation_results/cache", help="Cache directory")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (default: all)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1,3,5,10,20], help="K values")

    args = parser.parse_args()

    pipeline = EvaluationPipeline(args.db_path, args.cache_dir)

    results = pipeline.run_full_evaluation(
        models=args.models,
        k_values=args.k_values,
        use_cache=not args.no_cache
    )

    pipeline.print_comparison(results)
