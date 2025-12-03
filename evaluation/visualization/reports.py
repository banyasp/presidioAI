"""Report generation for evaluation results."""

import pandas as pd
from typing import Dict, List
import os


def generate_comparison_table(results: Dict, format: str = "markdown") -> str:
    """
    Generate a comparison table of model performance.

    Args:
        results: Results dictionary from run_full_evaluation()
        format: Output format ("markdown", "latex", "html")

    Returns:
        Formatted table string
    """
    # Extract metrics for each model
    rows = []

    for model_name, model_results in results['models'].items():
        metrics = model_results['aggregate_metrics']

        row = {
            'Model': model_name,
            'P@1': f"{metrics.get('precision@1', 0):.4f}",
            'P@3': f"{metrics.get('precision@3', 0):.4f}",
            'P@5': f"{metrics.get('precision@5', 0):.4f}",
            'P@10': f"{metrics.get('precision@10', 0):.4f}",
            'MRR': f"{metrics.get('mrr', 0):.4f}",
            'NDCG@3': f"{metrics.get('ndcg@3', 0):.4f}",
            'NDCG@5': f"{metrics.get('ndcg@5', 0):.4f}",
            'NDCG@10': f"{metrics.get('ndcg@10', 0):.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if format == "markdown":
        return df.to_markdown(index=False)
    elif format == "latex":
        return df.to_latex(index=False)
    elif format == "html":
        return df.to_html(index=False)
    else:
        return str(df)


def generate_text_report(results: Dict, output_path: str):
    """
    Generate a comprehensive text report.

    Args:
        results: Results dictionary from run_full_evaluation()
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Run ID: {results['run_id']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Models evaluated: {len(results['models'])}\n\n")

        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write(generate_comparison_table(results, format="markdown"))
        f.write("\n\n")

        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS PER MODEL\n")
        f.write("="*80 + "\n\n")

        for model_name, model_results in results['models'].items():
            f.write(f"\n{model_name}\n")
            f.write("-" * len(model_name) + "\n\n")

            metrics = model_results['aggregate_metrics']
            f.write(f"Number of queries: {model_results['num_queries']}\n\n")

            f.write("Precision@k:\n")
            for k in [1, 3, 5, 10, 20]:
                key = f'precision@{k}'
                if key in metrics:
                    f.write(f"  P@{k:2d}: {metrics[key]:.4f}\n")

            f.write("\nRecall@k:\n")
            for k in [1, 3, 5, 10, 20]:
                key = f'recall@{k}'
                if key in metrics:
                    f.write(f"  R@{k:2d}: {metrics[key]:.4f}\n")

            f.write("\nNDCG@k:\n")
            for k in [1, 3, 5, 10, 20]:
                key = f'ndcg@{k}'
                if key in metrics:
                    f.write(f"  NDCG@{k:2d}: {metrics[key]:.4f}\n")

            f.write(f"\nMean Reciprocal Rank: {metrics.get('mrr', 0):.4f}\n")

    print(f"✓ Report saved to: {output_path}")


def perform_error_analysis(per_query_metrics_df: pd.DataFrame, output_path: str, threshold: float = 0.3):
    """
    Perform error analysis and save challenging queries.

    Args:
        per_query_metrics_df: Per-query metrics DataFrame
        output_path: Path to output CSV file
        threshold: NDCG threshold below which queries are considered "challenging"
    """
    # Identify challenging queries (low NDCG across all models)
    if 'ndcg@10' in per_query_metrics_df.columns:
        challenging = per_query_metrics_df[per_query_metrics_df['ndcg@10'] < threshold]

        if not challenging.empty:
            challenging.to_csv(output_path, index=False)
            print(f"✓ Error analysis saved to: {output_path}")
            print(f"  Found {len(challenging)} challenging queries (NDCG@10 < {threshold})")
        else:
            print(f"  No challenging queries found (all NDCG@10 >= {threshold})")


def compute_statistical_significance(model1_metrics: pd.DataFrame,
                                     model2_metrics: pd.DataFrame,
                                     metric: str = 'ndcg@10') -> Dict:
    """
    Compute statistical significance between two models using paired t-test.

    Args:
        model1_metrics: Per-query metrics for model 1
        model2_metrics: Per-query metrics for model 2
        metric: Metric to compare

    Returns:
        Dictionary with test statistics
    """
    from scipy import stats

    if metric not in model1_metrics.columns or metric not in model2_metrics.columns:
        return {'error': f'Metric {metric} not found'}

    # Ensure same queries
    if not (model1_metrics['query_id'] == model2_metrics['query_id']).all():
        return {'error': 'Query IDs do not match'}

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        model1_metrics[metric],
        model2_metrics[metric]
    )

    return {
        'metric': metric,
        'mean_diff': model1_metrics[metric].mean() - model2_metrics[metric].mean(),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


if __name__ == "__main__":
    # Test report generation
    print("Testing report generation...")

    mock_results = {
        'run_id': '12345',
        'timestamp': '2025-01-01T00:00:00',
        'models': {
            'legal-bert': {
                'aggregate_metrics': {
                    'precision@1': 0.85,
                    'precision@3': 0.72,
                    'precision@5': 0.65,
                    'precision@10': 0.58,
                    'mrr': 0.89,
                    'ndcg@3': 0.76,
                    'ndcg@5': 0.71,
                    'ndcg@10': 0.68
                },
                'num_queries': 345
            },
            'harvard-bert': {
                'aggregate_metrics': {
                    'precision@1': 0.82,
                    'precision@3': 0.70,
                    'precision@5': 0.63,
                    'precision@10': 0.56,
                    'mrr': 0.87,
                    'ndcg@3': 0.74,
                    'ndcg@5': 0.69,
                    'ndcg@10': 0.66
                },
                'num_queries': 345
            }
        }
    }

    print("\nMarkdown table:")
    print(generate_comparison_table(mock_results))

    print("\n✓ Report generation tests complete")
