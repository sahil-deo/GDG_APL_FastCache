import pytest
from fastcache import SemanticCache
from fastcache.embedders.base import BaseEmbedder
import numpy as np

class DummyEmbedder(BaseEmbedder):
    @property
    def dimension(self):
        return 2
        
    def embed(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

def test_cache_hit_and_miss():
    cache = SemanticCache(embedder=DummyEmbedder())
    
    calls = 0
    def fallback(prompt):
        nonlocal calls
        calls += 1
        return "response"
        
    res1 = cache.query("hello", fallback)
    assert res1 == "response"
    assert calls == 1
    
    res2 = cache.query("hello", fallback)
    assert res2 == "response"
    assert calls == 1  # Hit exact match fast path
    
    assert cache.stats.total_queries == 2
    assert cache.stats.cache_hits == 1
    assert cache.stats.cache_misses == 1
