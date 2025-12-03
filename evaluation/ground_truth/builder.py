"""Build ground truth from case citations with fuzzy matching."""

import json
import sqlite3
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher
import logging
from tqdm import tqdm

from ..storage.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fuzzy_match_case_name(target_name: str, all_cases: List[Dict], threshold: float = 0.85) -> Optional[int]:
    """
    Match a case name to a case ID using fuzzy matching.

    Args:
        target_name: Case name to match (e.g., from citations)
        all_cases: List of all cases with 'id' and 'case_name' keys
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        Case ID if match found, None otherwise
    """
    if not target_name:
        return None

    # Clean up target name
    target_clean = target_name.strip().lower()

    best_match_id = None
    best_score = 0.0

    for case in all_cases:
        case_name = case['case_name'].strip().lower()

        # Exact match (case-insensitive)
        if target_clean == case_name:
            return case['id']

        # Substring match (one is contained in the other)
        if target_clean in case_name or case_name in target_clean:
            # Give bonus for substring matches
            score = max(len(target_clean) / len(case_name), len(case_name) / len(target_clean)) * 1.1
            if score > best_score:
                best_score = score
                best_match_id = case['id']
        else:
            # Fuzzy string similarity
            similarity = SequenceMatcher(None, target_clean, case_name).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match_id = case['id']

    # Return match only if above threshold
    if best_score >= threshold:
        return best_match_id

    return None


def build_ground_truth_for_query(
    query_id: int,
    source_case_id: int,
    related_cases_json: str,
    all_cases: List[Dict],
    source_relevance: int = 2,
    citation_relevance: int = 1,
    match_threshold: float = 0.85
) -> List[Tuple[int, int]]:
    """
    Build ground truth relevance scores for a single query.

    Args:
        query_id: Query ID
        source_case_id: ID of case the query was generated from
        related_cases_json: JSON string of related case names
        all_cases: List of all cases for fuzzy matching
        source_relevance: Relevance score for source case (default: 2)
        citation_relevance: Relevance score for cited cases (default: 1)
        match_threshold: Fuzzy matching threshold

    Returns:
        List of (case_id, relevance_score) tuples
    """
    ground_truth = []

    # Source case always has highest relevance
    ground_truth.append((source_case_id, source_relevance))

    # Parse related cases
    if not related_cases_json or related_cases_json == "null":
        return ground_truth

    try:
        related_cases = json.loads(related_cases_json)
        if not isinstance(related_cases, list):
            logger.warning(f"Query {query_id}: related_cases is not a list")
            return ground_truth

        # Match each cited case to a case ID
        for cited_case_name in related_cases:
            if not cited_case_name or not isinstance(cited_case_name, str):
                continue

            matched_case_id = fuzzy_match_case_name(cited_case_name, all_cases, threshold=match_threshold)

            if matched_case_id:
                # Don't duplicate if it's the source case
                if matched_case_id != source_case_id:
                    ground_truth.append((matched_case_id, citation_relevance))
            # Silently skip unmatched citations (there are many)

    except json.JSONDecodeError as e:
        logger.warning(f"Query {query_id}: Failed to parse related_cases JSON: {e}")

    return ground_truth


def populate_ground_truth_table(
    db_path: str,
    source_relevance: int = 2,
    citation_relevance: int = 1,
    match_threshold: float = 0.85
) -> Dict[str, int]:
    """
    Populate the evaluation_ground_truth table for all queries.

    Args:
        db_path: Path to SQLite database
        source_relevance: Relevance score for source cases
        citation_relevance: Relevance score for cited cases
        match_threshold: Fuzzy matching threshold

    Returns:
        Dictionary with statistics
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Load all cases for fuzzy matching
    cursor.execute("SELECT id, case_name FROM cases")
    all_cases = [{'id': row['id'], 'case_name': row['case_name']} for row in cursor.fetchall()]

    logger.info(f"Loaded {len(all_cases)} cases for fuzzy matching")

    # Load all queries with their source cases
    cursor.execute("""
        SELECT
            eq.id as query_id,
            eq.source_case_id,
            c.related_cases
        FROM evaluation_queries eq
        JOIN cases c ON eq.source_case_id = c.id
    """)

    queries = cursor.fetchall()
    logger.info(f"Processing {len(queries)} queries")

    stats = {
        'total_queries': len(queries),
        'total_ground_truth_entries': 0,
        'queries_with_citations': 0,
        'total_citations_matched': 0,
        'avg_citations_per_query': 0.0
    }

    citation_counts = []

    for query_row in tqdm(queries, desc="Building ground truth"):
        query_id = query_row['query_id']
        source_case_id = query_row['source_case_id']
        related_cases_json = query_row['related_cases']

        # Build ground truth for this query
        ground_truth = build_ground_truth_for_query(
            query_id,
            source_case_id,
            related_cases_json,
            all_cases,
            source_relevance=source_relevance,
            citation_relevance=citation_relevance,
            match_threshold=match_threshold
        )

        # Insert into database
        for case_id, relevance in ground_truth:
            cursor.execute("""
                INSERT OR IGNORE INTO evaluation_ground_truth
                    (query_id, case_id, relevance_score)
                VALUES (?, ?, ?)
            """, (query_id, case_id, relevance))

        # Update stats
        stats['total_ground_truth_entries'] += len(ground_truth)
        citation_count = len(ground_truth) - 1  # Exclude source case
        if citation_count > 0:
            stats['queries_with_citations'] += 1
            citation_counts.append(citation_count)
            stats['total_citations_matched'] += citation_count

    if citation_counts:
        stats['avg_citations_per_query'] = sum(citation_counts) / len(citation_counts)

    conn.commit()
    conn.close()

    logger.info(f"✓ Built ground truth for {stats['total_queries']} queries")
    logger.info(f"  Total ground truth entries: {stats['total_ground_truth_entries']}")
    logger.info(f"  Queries with citations: {stats['queries_with_citations']}")
    logger.info(f"  Total citations matched: {stats['total_citations_matched']}")
    logger.info(f"  Avg citations per query: {stats['avg_citations_per_query']:.2f}")

    return stats


def get_ground_truth_for_query(db_path: str, query_id: int) -> Dict[int, int]:
    """
    Get ground truth relevance scores for a specific query.

    Args:
        db_path: Path to SQLite database
        query_id: Query ID

    Returns:
        Dictionary mapping case_id -> relevance_score
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT case_id, relevance_score
        FROM evaluation_ground_truth
        WHERE query_id = ?
    """, (query_id,))

    ground_truth = {row['case_id']: row['relevance_score'] for row in cursor.fetchall()}

    conn.close()
    return ground_truth


def print_sample_ground_truth(db_path: str, n: int = 5) -> None:
    """
    Print sample ground truth for inspection.

    Args:
        db_path: Path to SQLite database
        n: Number of samples to show
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            eq.id as query_id,
            eq.query_text,
            eq.source_case_id,
            c.case_name as source_case_name
        FROM evaluation_queries eq
        JOIN cases c ON eq.source_case_id = c.id
        LIMIT ?
    """, (n,))

    queries = cursor.fetchall()

    print(f"\n{'='*80}")
    print("SAMPLE GROUND TRUTH")
    print(f"{'='*80}\n")

    for query_row in queries:
        query_id = query_row['query_id']
        query_text = query_row['query_text']
        source_case_name = query_row['source_case_name']

        print(f"Query #{query_id}:")
        print(f"  Text: {query_text[:100]}...")
        print(f"  Source case: {source_case_name}")

        # Get ground truth
        cursor.execute("""
            SELECT
                gt.case_id,
                gt.relevance_score,
                c.case_name
            FROM evaluation_ground_truth gt
            JOIN cases c ON gt.case_id = c.id
            WHERE gt.query_id = ?
            ORDER BY gt.relevance_score DESC
        """, (query_id,))

        ground_truth = cursor.fetchall()

        print(f"  Ground truth ({len(ground_truth)} cases):")
        for gt_row in ground_truth[:10]:  # Show first 10
            case_name = gt_row['case_name']
            relevance = gt_row['relevance_score']
            marker = "SOURCE" if relevance == 2 else "CITED"
            print(f"    [{marker}] {case_name} (relevance={relevance})")

        if len(ground_truth) > 10:
            print(f"    ... and {len(ground_truth) - 10} more")

        print()

    conn.close()

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python builder.py <path_to_db>")
        sys.exit(1)

    db_path = sys.argv[1]

    print("Building ground truth...")
    stats = populate_ground_truth_table(db_path)

    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSamples:")
    print_sample_ground_truth(db_path, n=3)
