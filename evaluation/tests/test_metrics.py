"""Unit tests for evaluation metrics."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.metrics.compute import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    dcg_at_k,
    idcg_at_k
)


def test_precision_at_k():
    """Test Precision@k calculation."""
    ground_truth = {1: 2, 3: 1}  # case 1 highly relevant, case 3 relevant
    # Ideal@k for this ground truth:
    # - k=1: ideal=2 (just source)
    # - k=2: ideal=2+1=3 (source + 1 citation)
    # - k=3: ideal=2+1=3 (only 2 relevant cases exist)

    # Perfect ranking
    results = [1, 3, 2, 4, 5]
    # top-3: [1(rel=2), 3(rel=1), 2(rel=0)] = sum=3
    # ideal: 2+1 = 3 (both relevant cases)
    assert precision_at_k(results, ground_truth, 3) == 1.0  # 3/3 = 1.0

    # Good ranking (swapped order)
    results = [3, 1, 2]
    # top-3: [3(rel=1), 1(rel=2), 2(rel=0)] = sum=3
    # ideal: 3 (both relevant cases)
    assert precision_at_k(results, ground_truth, 3) == 1.0  # 3/3 = 1.0

    # Partial ranking (only one relevant in top-3)
    results = [1, 2, 4]
    # top-3: [1(rel=2), 2(rel=0), 4(rel=0)] = sum=2
    # ideal: 3 (both relevant cases)
    assert abs(precision_at_k(results, ground_truth, 3) - 0.667) < 0.01  # 2/3 ≈ 0.667

    # Poor ranking (no relevant in top-3)
    results = [2, 4, 5]
    assert precision_at_k(results, ground_truth, 3) == 0.0

    print("✓ Precision@k tests passed")


def test_recall_at_k():
    """Test Recall@k calculation."""
    ground_truth = {1: 2, 3: 1}  # 2 relevant cases

    # Both relevant in top-3
    results = [1, 3, 2]
    assert recall_at_k(results, ground_truth, 3) == 1.0

    # One relevant in top-2
    results = [1, 2, 3]
    assert recall_at_k(results, ground_truth, 2) == 0.5

    # No relevant in top-1
    results = [2, 1, 3]
    assert recall_at_k(results, ground_truth, 1) == 0.0

    print("✓ Recall@k tests passed")


def test_mean_reciprocal_rank():
    """Test MRR calculation."""
    ground_truth = {1: 2, 3: 1}

    # First result is relevant
    results = [1, 2, 3]
    assert mean_reciprocal_rank(results, ground_truth) == 1.0

    # Second result is relevant
    results = [2, 3, 1]
    assert mean_reciprocal_rank(results, ground_truth) == 0.5

    # Third result is relevant
    results = [2, 4, 1]
    assert abs(mean_reciprocal_rank(results, ground_truth) - 0.3333) < 0.001

    print("✓ MRR tests passed")


def test_ndcg_at_k():
    """Test NDCG@k calculation."""
    ground_truth = {1: 2, 3: 1}

    # Perfect ranking (highest relevance first)
    results = [1, 3, 2]
    assert ndcg_at_k(results, ground_truth, 3) == 1.0

    # Good ranking (both relevant but swapped)
    results = [3, 1, 2]
    ndcg = ndcg_at_k(results, ground_truth, 3)
    assert 0.85 < ndcg < 0.87  # Should be slightly less than 1.0

    # Poor ranking (relevant items late)
    results = [2, 4, 1, 3]
    ndcg = ndcg_at_k(results, ground_truth, 3)
    assert ndcg < 0.5

    print("✓ NDCG@k tests passed")


def test_dcg_and_idcg():
    """Test DCG and IDCG calculation."""
    relevances = [2, 1, 0]

    # DCG with perfect ordering
    dcg = dcg_at_k(relevances, 3)
    assert dcg > 0

    # IDCG should equal DCG for already-sorted list
    idcg = idcg_at_k(relevances, 3)
    assert abs(dcg - idcg) < 0.001

    # IDCG for unsorted should be higher than DCG
    unsorted = [0, 1, 2]
    dcg_unsorted = dcg_at_k(unsorted, 3)
    idcg_unsorted = idcg_at_k(unsorted, 3)
    assert idcg_unsorted > dcg_unsorted

    print("✓ DCG/IDCG tests passed")


if __name__ == "__main__":
    test_precision_at_k()
    test_recall_at_k()
    test_mean_reciprocal_rank()
    test_ndcg_at_k()
    test_dcg_and_idcg()

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
