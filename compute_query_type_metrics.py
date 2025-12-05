#!/usr/bin/env python3
"""Compute metrics separated by query type using cached predictions."""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def compute_precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Compute precision@k."""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([doc_id for doc_id in retrieved_at_k if doc_id in relevant])
    return relevant_retrieved / k

def compute_mrr(retrieved: List[int], relevant: List[int]) -> float:
    """Compute mean reciprocal rank."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def compute_ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Compute NDCG@k."""
    if k == 0 or not relevant:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / (i.bit_length())  # log2(i+1)
    
    # IDCG@k
    idcg = sum(1.0 / (i.bit_length()) for i in range(1, min(len(relevant), k) + 1))
    
    return dcg / idcg if idcg > 0 else 0.0

def load_cached_predictions(cache_dir: str, model_name: str) -> Dict[int, List[int]]:
    """Load cached predictions for a model."""
    model_cache_dir = Path(cache_dir) / model_name
    predictions = {}
    
    if not model_cache_dir.exists():
        print(f"Warning: Cache directory not found: {model_cache_dir}")
        return predictions
    
    for json_file in model_cache_dir.glob("*.json"):
        query_id = int(json_file.stem)
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Predictions are stored as list of case IDs in ranked order
            # The key is 'results' not 'predictions'
            predictions[query_id] = data.get('results', data.get('predictions', []))
    
    print(f"Loaded {len(predictions)} cached predictions for {model_name}")
    return predictions

def get_ground_truth(db_path: str) -> Dict[int, List[int]]:
    """Get ground truth relevant cases for each query."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT query_id, case_id, relevance_score
        FROM evaluation_ground_truth
        ORDER BY query_id, relevance_score DESC
    """)
    
    ground_truth = defaultdict(list)
    for row in cursor.fetchall():
        query_id = row['query_id']
        case_id = row['case_id']
        ground_truth[query_id].append(case_id)
    
    conn.close()
    return dict(ground_truth)

def get_query_types(db_path: str) -> Dict[int, str]:
    """Get query types for all queries."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, query_type FROM evaluation_queries")
    query_types = {row['id']: row['query_type'] for row in cursor.fetchall()}
    
    conn.close()
    return query_types

def compute_metrics_by_query_type(db_path: str, cache_dir: str, models: List[str], k_values: List[int]):
    """Compute metrics for each model, separated by query type."""
    
    print("Loading data...")
    ground_truth = get_ground_truth(db_path)
    query_types = get_query_types(db_path)
    
    results = {}
    
    for model in models:
        print(f"\nProcessing {model}...")
        predictions = load_cached_predictions(cache_dir, model)
        
        # Separate queries by type
        metrics_by_type = {
            'extractive': defaultdict(list),
            'generative': defaultdict(list),
            'overall': defaultdict(list)
        }
        
        for query_id, predicted_cases in predictions.items():
            if query_id not in ground_truth:
                continue
            
            relevant_cases = ground_truth[query_id]
            query_type = query_types.get(query_id, 'unknown')
            
            if query_type not in ['extractive', 'generative']:
                continue
            
            # Compute metrics for this query
            for k in k_values:
                p_at_k = compute_precision_at_k(predicted_cases, relevant_cases, k)
                ndcg_at_k = compute_ndcg_at_k(predicted_cases, relevant_cases, k)
                
                metrics_by_type[query_type][f'precision@{k}'].append(p_at_k)
                metrics_by_type[query_type][f'ndcg@{k}'].append(ndcg_at_k)
                metrics_by_type['overall'][f'precision@{k}'].append(p_at_k)
                metrics_by_type['overall'][f'ndcg@{k}'].append(ndcg_at_k)
            
            mrr = compute_mrr(predicted_cases, relevant_cases)
            metrics_by_type[query_type]['mrr'].append(mrr)
            metrics_by_type['overall']['mrr'].append(mrr)
        
        # Average metrics
        model_results = {}
        for query_type in ['extractive', 'generative', 'overall']:
            model_results[query_type] = {}
            for metric_name, values in metrics_by_type[query_type].items():
                if values:
                    model_results[query_type][metric_name] = sum(values) / len(values)
                    model_results[query_type][f'{metric_name}_count'] = len(values)
        
        results[model] = model_results
    
    return results

def print_results_table(results: Dict):
    """Print results in a formatted table."""
    models = list(results.keys())
    
    for query_type in ['extractive', 'generative', 'overall']:
        print(f"\n{'='*80}")
        print(f"{query_type.upper()} QUERIES")
        print(f"{'='*80}\n")
        
        # Header
        print(f"{'Model':<25} {'P@1':>8} {'P@3':>8} {'P@10':>8} {'MRR':>8} {'NDCG@3':>8} {'NDCG@10':>8}")
        print("-" * 80)
        
        for model in models:
            if query_type not in results[model]:
                continue
            metrics = results[model][query_type]
            
            p1 = metrics.get('precision@1', 0)
            p3 = metrics.get('precision@3', 0)
            p10 = metrics.get('precision@10', 0)
            mrr = metrics.get('mrr', 0)
            ndcg3 = metrics.get('ndcg@3', 0)
            ndcg10 = metrics.get('ndcg@10', 0)
            
            print(f"{model:<25} {p1:>8.4f} {p3:>8.4f} {p10:>8.4f} {mrr:>8.4f} {ndcg3:>8.4f} {ndcg10:>8.4f}")
        
        # Print query counts
        if models:
            first_model = models[0]
            if query_type in results[first_model]:
                count = results[first_model][query_type].get('mrr_count', 0)
                print(f"\nNumber of queries: {count}")

def save_results_to_file(results: Dict, output_path: str):
    """Save results to a markdown file."""
    lines = []
    
    lines.append("# Model Performance by Query Type\n")
    lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for query_type in ['extractive', 'generative', 'overall']:
        lines.append(f"\n## {query_type.title()} Queries\n")
        
        lines.append("| Model | P@1 | P@3 | P@10 | MRR | NDCG@3 | NDCG@10 |")
        lines.append("|-------|-----|-----|------|-----|--------|---------|")
        
        for model in results:
            if query_type not in results[model]:
                continue
            metrics = results[model][query_type]
            
            p1 = metrics.get('precision@1', 0)
            p3 = metrics.get('precision@3', 0)
            p10 = metrics.get('precision@10', 0)
            mrr = metrics.get('mrr', 0)
            ndcg3 = metrics.get('ndcg@3', 0)
            ndcg10 = metrics.get('ndcg@10', 0)
            
            lines.append(f"| {model} | {p1:.4f} | {p3:.4f} | {p10:.4f} | {mrr:.4f} | {ndcg3:.4f} | {ndcg10:.4f} |")
        
        # Add query count
        if results:
            first_model = list(results.keys())[0]
            if query_type in results[first_model]:
                count = results[first_model][query_type].get('mrr_count', 0)
                lines.append(f"\n**Number of queries**: {count}\n")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "scotus_cases.db"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "evaluation_results/cache"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "evaluation_results/reports/query_type_breakdown.md"
    
    models = ['sentence-transformer', 'legal-bert', 'harvard-bert']
    k_values = [1, 3, 5, 10, 20]
    
    results = compute_metrics_by_query_type(db_path, cache_dir, models, k_values)
    print_results_table(results)
    save_results_to_file(results, output_file)
