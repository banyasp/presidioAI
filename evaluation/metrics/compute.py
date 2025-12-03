"""Information retrieval metrics: Precision@k, MRR, NDCG@k."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(results: List[int], ground_truth: Dict[int, int], k: int) -> float:
    """
    Compute Precision@k with graded relevance.

    For graded relevance, the ideal maximum is computed from the actual ground truth.
    Since there is only ONE source case (relevance=2) per query, the ideal maximum is:
    - Position 1: Source case (relevance=2)
    - Positions 2 to k: Cited cases (relevance=1 each)
    - Ideal max = 2 + (k-1) = k + 1 (assuming enough citations exist)

    More generally: ideal = sum of top-k relevances from sorted ground truth.

    Args:
        results: List of case IDs ranked by similarity (best first)
        ground_truth: Dict mapping case_id -> relevance_score
        k: Number of top results to consider

    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not results or k <= 0:
        return 0.0

    # Get top-k results
    top_k_results = results[:k]

    # Sum relevance scores of top-k results
    relevance_sum = sum(ground_truth.get(case_id, 0) for case_id in top_k_results)

    # Ideal maximum: sum of top-k relevances from ground truth (sorted descending)
    sorted_relevances = sorted(ground_truth.values(), reverse=True)
    ideal_max = sum(sorted_relevances[:k])

    # Avoid division by zero
    if ideal_max == 0:
        return 0.0

    # Normalize by ideal maximum
    precision = relevance_sum / ideal_max

    return precision


def mean_reciprocal_rank(results: List[int], ground_truth: Dict[int, int], min_relevance: int = 1) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_result

    Args:
        results: List of case IDs ranked by similarity (best first)
        ground_truth: Dict mapping case_id -> relevance_score
        min_relevance: Minimum relevance score to consider a result "relevant"

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not results:
        return 0.0

    for rank, case_id in enumerate(results, start=1):
        relevance = ground_truth.get(case_id, 0)
        if relevance >= min_relevance:
            return 1.0 / rank

    return 0.0  # No relevant result found


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    DCG@k = sum(rel_i / log2(i+1)) for i in 1..k

    Args:
        relevances: List of relevance scores (in ranking order)
        k: Number of top results to consider

    Returns:
        DCG@k score
    """
    if not relevances or k <= 0:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)

    return dcg


def idcg_at_k(ground_truth_scores: List[int], k: int) -> float:
    """
    Compute Ideal DCG at k (DCG with perfect ranking).

    IDCG@k = DCG@k of relevance scores sorted in descending order

    Args:
        ground_truth_scores: All relevance scores for this query
        k: Number of top results to consider

    Returns:
        IDCG@k score
    """
    if not ground_truth_scores:
        return 0.0

    # Sort in descending order (ideal ranking)
    sorted_scores = sorted(ground_truth_scores, reverse=True)

    return dcg_at_k(sorted_scores, k)


def ndcg_at_k(results: List[int], ground_truth: Dict[int, int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k

    Args:
        results: List of case IDs ranked by similarity (best first)
        ground_truth: Dict mapping case_id -> relevance_score
        k: Number of top results to consider

    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if not results or k <= 0:
        return 0.0

    # Get relevance scores for results
    relevances = [ground_truth.get(case_id, 0) for case_id in results[:k]]

    # Compute DCG@k
    dcg = dcg_at_k(relevances, k)

    # Compute IDCG@k (ideal ranking)
    all_relevances = list(ground_truth.values())
    idcg = idcg_at_k(all_relevances, k)

    # Avoid division by zero
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def recall_at_k(results: List[int], ground_truth: Dict[int, int], k: int, min_relevance: int = 1) -> float:
    """
    Compute Recall@k.

    Recall@k = (# relevant docs in top-k) / (total # relevant docs)

    Args:
        results: List of case IDs ranked by similarity (best first)
        ground_truth: Dict mapping case_id -> relevance_score
        k: Number of top results to consider
        min_relevance: Minimum relevance score to consider a result "relevant"

    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not results or k <= 0:
        return 0.0

    # Get top-k results
    top_k_results = set(results[:k])

    # Count relevant docs in top-k
    relevant_in_top_k = sum(1 for case_id in top_k_results if ground_truth.get(case_id, 0) >= min_relevance)

    # Count total relevant docs
    total_relevant = sum(1 for rel in ground_truth.values() if rel >= min_relevance)

    if total_relevant == 0:
        return 0.0

    return relevant_in_top_k / total_relevant


class MetricsAggregator:
    """Accumulate per-query metrics and compute aggregates."""

    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Initialize metrics aggregator.

        Args:
            k_values: List of k values for Precision@k and NDCG@k (default: [1, 3, 5, 10, 20])
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.per_query_metrics = []
        self.query_ids = []

    def add_query_result(self, query_id: int, results: List[int], ground_truth: Dict[int, int]) -> Dict[str, float]:
        """
        Compute and store metrics for a single query.

        Args:
            query_id: Query ID
            results: List of case IDs ranked by similarity
            ground_truth: Dict mapping case_id -> relevance_score

        Returns:
            Dictionary of metrics for this query
        """
        metrics = {}

        # Precision@k for each k
        for k in self.k_values:
            metrics[f'precision@{k}'] = precision_at_k(results, ground_truth, k)

        # Recall@k for each k
        for k in self.k_values:
            metrics[f'recall@{k}'] = recall_at_k(results, ground_truth, k)

        # NDCG@k for each k
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = ndcg_at_k(results, ground_truth, k)

        # MRR
        metrics['mrr'] = mean_reciprocal_rank(results, ground_truth)

        # Store
        self.per_query_metrics.append(metrics)
        self.query_ids.append(query_id)

        return metrics

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute mean metrics across all queries.

        Returns:
            Dictionary of aggregate metrics
        """
        if not self.per_query_metrics:
            return {}

        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(self.per_query_metrics)

        # Compute means
        return df.mean().to_dict()

    def get_per_query_dataframe(self) -> pd.DataFrame:
        """
        Get per-query metrics as a DataFrame.

        Returns:
            DataFrame with query_id and all metrics
        """
        if not self.per_query_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(self.per_query_metrics)
        df.insert(0, 'query_id', self.query_ids)

        return df

    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics (mean, std, min, max) for each metric.

        Returns:
            Nested dictionary: {metric_name: {stat_name: value}}
        """
        df = self.get_per_query_dataframe()

        if df.empty:
            return {}

        # Drop query_id column
        metric_df = df.drop('query_id', axis=1)

        stats = {}
        for col in metric_df.columns:
            stats[col] = {
                'mean': metric_df[col].mean(),
                'std': metric_df[col].std(),
                'min': metric_df[col].min(),
                'max': metric_df[col].max(),
                'median': metric_df[col].median()
            }

        return stats

    def print_summary(self):
        """Print a summary of aggregate metrics."""
        agg_metrics = self.get_aggregate_metrics()

        print(f"\n{'='*80}")
        print("AGGREGATE METRICS")
        print(f"{'='*80}")
        print(f"Queries evaluated: {len(self.per_query_metrics)}")
        print()

        # Group by metric type
        precision_metrics = {k: v for k, v in agg_metrics.items() if k.startswith('precision@')}
        recall_metrics = {k: v for k, v in agg_metrics.items() if k.startswith('recall@')}
        ndcg_metrics = {k: v for k, v in agg_metrics.items() if k.startswith('ndcg@')}
        mrr = agg_metrics.get('mrr', 0.0)

        print("Precision@k:")
        for k in self.k_values:
            metric_name = f'precision@{k}'
            if metric_name in precision_metrics:
                print(f"  P@{k:2d}: {precision_metrics[metric_name]:.4f}")

        print()
        print("Recall@k:")
        for k in self.k_values:
            metric_name = f'recall@{k}'
            if metric_name in recall_metrics:
                print(f"  R@{k:2d}: {recall_metrics[metric_name]:.4f}")

        print()
        print("NDCG@k:")
        for k in self.k_values:
            metric_name = f'ndcg@{k}'
            if metric_name in ndcg_metrics:
                print(f"  NDCG@{k:2d}: {ndcg_metrics[metric_name]:.4f}")

        print()
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the metrics with example data
    print("Testing metrics...")

    # Example: 5 documents, 2 relevant (case_id 1 with rel=2, case_id 3 with rel=1)
    ground_truth = {
        1: 2,  # Highly relevant
        3: 1   # Relevant
    }

    # Perfect ranking
    results_perfect = [1, 3, 2, 4, 5]
    print("\nPerfect ranking:", results_perfect)
    print(f"  Precision@3: {precision_at_k(results_perfect, ground_truth, 3):.4f}")
    print(f"  Recall@3: {recall_at_k(results_perfect, ground_truth, 3):.4f}")
    print(f"  NDCG@3: {ndcg_at_k(results_perfect, ground_truth, 3):.4f}")
    print(f"  MRR: {mean_reciprocal_rank(results_perfect, ground_truth):.4f}")

    # Good ranking (relevant but not perfect order)
    results_good = [3, 1, 2, 4, 5]
    print("\nGood ranking:", results_good)
    print(f"  Precision@3: {precision_at_k(results_good, ground_truth, 3):.4f}")
    print(f"  Recall@3: {recall_at_k(results_good, ground_truth, 3):.4f}")
    print(f"  NDCG@3: {ndcg_at_k(results_good, ground_truth, 3):.4f}")
    print(f"  MRR: {mean_reciprocal_rank(results_good, ground_truth):.4f}")

    # Poor ranking
    results_poor = [2, 4, 1, 3, 5]
    print("\nPoor ranking:", results_poor)
    print(f"  Precision@3: {precision_at_k(results_poor, ground_truth, 3):.4f}")
    print(f"  Recall@3: {recall_at_k(results_poor, ground_truth, 3):.4f}")
    print(f"  NDCG@3: {ndcg_at_k(results_poor, ground_truth, 3):.4f}")
    print(f"  MRR: {mean_reciprocal_rank(results_poor, ground_truth):.4f}")

    # Test aggregator
    print("\n" + "="*80)
    print("Testing MetricsAggregator...")
    print("="*80)

    aggregator = MetricsAggregator(k_values=[1, 3, 5])

    aggregator.add_query_result(1, results_perfect, ground_truth)
    aggregator.add_query_result(2, results_good, ground_truth)
    aggregator.add_query_result(3, results_poor, ground_truth)

    aggregator.print_summary()

    print("✓ Metrics tests complete")
