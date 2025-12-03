"""Caching system for evaluation results."""

import json
import os
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manage JSON-based caching for model predictions."""

    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (organized by model)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, model_name: str, query_id: int) -> str:
        """Get file path for cached result."""
        model_cache_dir = os.path.join(self.cache_dir, model_name)
        os.makedirs(model_cache_dir, exist_ok=True)
        return os.path.join(model_cache_dir, f"{query_id}.json")

    def get(self, model_name: str, query_id: int) -> Optional[List[int]]:
        """
        Retrieve cached results for a model and query.

        Args:
            model_name: Model name
            query_id: Query ID

        Returns:
            List of case IDs (ranked results) or None if not cached
        """
        cache_path = self._get_cache_path(model_name, query_id)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('results', [])
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache for {model_name}/query_{query_id}: {e}")
            return None

    def save(self, model_name: str, query_id: int, results: List[int]):
        """
        Save results to cache.

        Args:
            model_name: Model name
            query_id: Query ID
            results: List of case IDs (ranked results)
        """
        cache_path = self._get_cache_path(model_name, query_id)

        try:
            with open(cache_path, 'w') as f:
                json.dump({'results': results}, f)
        except IOError as e:
            logger.error(f"Failed to save cache for {model_name}/query_{query_id}: {e}")

    def clear(self, model_name: Optional[str] = None):
        """
        Clear cache for a specific model or all models.

        Args:
            model_name: Model name to clear, or None to clear all
        """
        if model_name:
            model_cache_dir = os.path.join(self.cache_dir, model_name)
            if os.path.exists(model_cache_dir):
                for file in os.listdir(model_cache_dir):
                    os.remove(os.path.join(model_cache_dir, file))
                logger.info(f"✓ Cleared cache for model: {model_name}")
        else:
            for model_dir in os.listdir(self.cache_dir):
                model_path = os.path.join(self.cache_dir, model_dir)
                if os.path.isdir(model_path):
                    for file in os.listdir(model_path):
                        os.remove(os.path.join(model_path, file))
            logger.info("✓ Cleared all caches")

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached results.

        Returns:
            Dictionary with cache stats per model
        """
        stats = {}

        if not os.path.exists(self.cache_dir):
            return stats

        for model_dir in os.listdir(self.cache_dir):
            model_path = os.path.join(self.cache_dir, model_dir)
            if os.path.isdir(model_path):
                cached_count = len([f for f in os.listdir(model_path) if f.endswith('.json')])
                stats[model_dir] = cached_count

        return stats


if __name__ == "__main__":
    # Test caching
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(tmpdir)

        # Test save and retrieve
        test_results = [1, 5, 3, 7, 2]
        cache.save("test-model", 42, test_results)

        retrieved = cache.get("test-model", 42)
        assert retrieved == test_results, "Cache retrieval failed"

        # Test non-existent
        assert cache.get("test-model", 999) is None

        # Test stats
        stats = cache.get_cache_stats()
        assert stats["test-model"] == 1

        # Test clear
        cache.clear("test-model")
        assert cache.get("test-model", 42) is None

        print("✓ Cache tests passed")
