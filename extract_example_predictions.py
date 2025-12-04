#!/usr/bin/env python3
"""Extract actual model predictions for specific examples."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List

def get_case_name(db_path: str, case_id: int) -> str:
    """Get case name for a case ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT case_name FROM cases WHERE id = ?", (case_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else f"Case {case_id}"

def get_queries_for_case(db_path: str, source_case_id: int) -> Dict:
    """Get both extractive and generative queries for a case."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, query_type, query_text
        FROM evaluation_queries
        WHERE source_case_id = ?
        ORDER BY query_type
    """, (source_case_id,))
    
    queries = {'extractive': [], 'generative': []}
    for row in cursor.fetchall():
        queries[row['query_type']].append({
            'id': row['id'],
            'text': row['query_text']
        })
    
    conn.close()
    return queries

def load_cached_prediction(cache_dir: str, model_name: str, query_id: int) -> List[int]:
    """Load cached prediction for a specific query."""
    cache_file = Path(cache_dir) / model_name / f"{query_id}.json"
    if not cache_file.exists():
        return []
    
    with open(cache_file, 'r') as f:
        data = json.load(f)
        return data.get('results', [])

def get_ground_truth_for_query(db_path: str, query_id: int) -> Dict[str, List[int]]:
    """Get ground truth for a query, categorized by relevance type."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get ground truth with relevance scores
    cursor.execute("""
        SELECT case_id, relevance_score
        FROM evaluation_ground_truth
        WHERE query_id = ?
        ORDER BY relevance_score DESC
    """, (query_id,))
    
    ground_truth = {'source': [], 'cited': []}
    for row in cursor.fetchall():
        if row['relevance_score'] == 2:  # Source case
            ground_truth['source'].append(row['case_id'])
        elif row['relevance_score'] == 1:  # Cited case
            ground_truth['cited'].append(row['case_id'])
    
    conn.close()
    return ground_truth

def categorize_relevance(case_id: int, ground_truth: Dict[str, List[int]]) -> str:
    """Categorize a case as Source, Cited, or Unrelated."""
    if case_id in ground_truth['source']:
        return "✓✓ Source"
    elif case_id in ground_truth['cited']:
        return "✓ Cited"
    else:
        return "✗ Unrelated"

def format_example(db_path: str, cache_dir: str, case_id: int, case_name: str, top_k: int = 3):
    """Format a complete example with model outputs."""
    queries = get_queries_for_case(db_path, case_id)
    
    if not queries['extractive'] or not queries['generative']:
        return f"### Example: {case_name}\n\n*No queries found for this case*\n"
    
    extractive_q = queries['extractive'][0]
    generative_q = queries['generative'][0]
    
    # Get ground truth for both queries
    ext_gt = get_ground_truth_for_query(db_path, extractive_q['id'])
    gen_gt = get_ground_truth_for_query(db_path, generative_q['id'])
    
    output = []
    output.append(f"### Example: {case_name}\n")
    
    # Extractive Query
    output.append(f"#### Extractive Query (Query ID: {extractive_q['id']})")
    output.append(f'> "{extractive_q["text"][:200]}..."')
    output.append(f"> \n> *Source: {case_name}*\n")
    
    # Generative Query
    output.append(f"#### Generative Query (Query ID: {generative_q['id']})")
    output.append(f'> "{generative_q["text"][:200]}..."\n')
    
    # Model outputs for extractive
    output.append(f"#### Model Results for Extractive Query\n")
    
    for model in ['sentence-transformer', 'legal-bert', 'harvard-bert']:
        predictions = load_cached_prediction(cache_dir, model, extractive_q['id'])
        output.append(f"**{model}**:")
        output.append("| Rank | Case ID | Case Name | Relevance |")
        output.append("|------|---------|-----------|-----------|")
        
        for rank, pred_case_id in enumerate(predictions[:top_k], 1):
            pred_case_name = get_case_name(db_path, pred_case_id)
            relevance = categorize_relevance(pred_case_id, ext_gt)
            output.append(f"| {rank} | {pred_case_id} | {pred_case_name} | {relevance} |")
        output.append("")
    
    # Model outputs for generative
    output.append(f"#### Model Results for Generative Query\n")
    
    for model in ['sentence-transformer', 'legal-bert', 'harvard-bert']:
        predictions = load_cached_prediction(cache_dir, model, generative_q['id'])
        output.append(f"**{model}**:")
        output.append("| Rank | Case ID | Case Name | Relevance |")
        output.append("|------|---------|-----------|-----------|")
        
        for rank, pred_case_id in enumerate(predictions[:top_k], 1):
            pred_case_name = get_case_name(db_path, pred_case_id)
            relevance = categorize_relevance(pred_case_id, gen_gt)
            output.append(f"| {rank} | {pred_case_id} | {pred_case_name} | {relevance} |")
        output.append("")
    
    output.append("**Analysis:**")
    
    # Check if sentence-transformer found it
    st_ext_pred = load_cached_prediction(cache_dir, 'sentence-transformer', extractive_q['id'])
    st_gen_pred = load_cached_prediction(cache_dir, 'sentence-transformer', generative_q['id'])
    
    st_ext_rank = st_ext_pred.index(case_id) + 1 if case_id in st_ext_pred else None
    st_gen_rank = st_gen_pred.index(case_id) + 1 if case_id in st_gen_pred else None
    
    if st_ext_rank == 1 and st_gen_rank == 1:
        output.append(f"- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types")
    elif st_ext_rank and st_gen_rank:
        output.append(f"- **sentence-transformer**: Found source at rank {st_ext_rank} (extractive) and rank {st_gen_rank} (generative)")
    else:
        output.append(f"- **sentence-transformer**: Failed to find source in top-100 for one or both queries")
    
    # Check legal-bert
    lb_ext_pred = load_cached_prediction(cache_dir, 'legal-bert', extractive_q['id'])
    lb_gen_pred = load_cached_prediction(cache_dir, 'legal-bert', generative_q['id'])
    
    lb_ext_rank = lb_ext_pred.index(case_id) + 1 if case_id in lb_ext_pred[:100] else None
    lb_gen_rank = lb_gen_pred.index(case_id) + 1 if case_id in lb_gen_pred[:100] else None
    
    if lb_ext_rank or lb_gen_rank:
        ranks = []
        if lb_ext_rank: ranks.append(f"rank {lb_ext_rank} (extractive)")
        if lb_gen_rank: ranks.append(f"rank {lb_gen_rank} (generative)")
        output.append(f"- **legal-bert**: Source found at {', '.join(ranks)}")
    else:
        output.append(f"- **legal-bert**: ❌ Failed to find source case in top-100 for both queries")
    
    # Check harvard-bert
    hb_ext_pred = load_cached_prediction(cache_dir, 'harvard-bert', extractive_q['id'])
    hb_gen_pred = load_cached_prediction(cache_dir, 'harvard-bert', generative_q['id'])
    
    hb_ext_rank = hb_ext_pred.index(case_id) + 1 if case_id in hb_ext_pred[:100] else None
    hb_gen_rank = hb_gen_pred.index(case_id) + 1 if case_id in hb_gen_pred[:100] else None
    
    if hb_ext_rank or hb_gen_rank:
        ranks = []
        if hb_ext_rank: ranks.append(f"rank {hb_ext_rank} (extractive)")
        if hb_gen_rank: ranks.append(f"rank {hb_gen_rank} (generative)")
        output.append(f"- **harvard-bert**: Source found at {', '.join(ranks)}")
    else:
        output.append(f"- **harvard-bert**: ❌ Failed to find source case in top-100 for both queries")
    
    output.append("\n---\n")
    
    return '\n'.join(output)

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "scotus_cases.db"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "evaluation_results/cache"
    
    # Example cases - diverse selection
    examples = [
        (1, "Trump v. CASA, Inc."),
        (2, "Kennedy v. Braidwood Management, Inc."),
        (5, "Free Speech Coalition, Inc. v. Paxton"),
        (66, "Loper Bright Enterprises v. Raimondo"),
        (112, "Trump v. Anderson"),
    ]
    
    print("## Concrete Examples with Model Outputs\n")
    print("Let's examine specific cases to see what each model actually returned. Results categorized as:")
    print("- **✓✓ Source**: The original source case")
    print("- **✓ Cited**: Cases cited in the source case's opinion (relevant)")
    print("- **✗ Unrelated**: Cases not relevant to the query")
    print("\n**Note**: P@5 = 0.552 is an AVERAGE across all queries. Individual queries vary - some retrieve more cited cases than others.\n")
    
    for case_id, case_name in examples:
        print(format_example(db_path, cache_dir, case_id, case_name, top_k=10))
