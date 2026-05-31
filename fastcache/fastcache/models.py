from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time

@dataclass
class CacheEntry:
    id: str                          # UUID4, auto-generated
    query: str                       # original prompt text
    response: str                    # cached LLM response
    vector: np.ndarray               # float32 embedding vector
    namespace: str                   # partition key
    created_at: float                # Unix timestamp
    ttl: int                         # seconds; 0 = no expiry
    hit_count: int = 0               # how many times this entry has been returned
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        if self.ttl == 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def ttl_remaining(self) -> float:
        if self.ttl == 0:
            return float("inf")
        remaining = self.ttl - self.age_seconds
        return max(0.0, remaining)

@dataclass
class LookupResult:
    hit: bool                        # True if similarity >= threshold
    similarity: float                # best cosine similarity found (0.0 if cache empty)
    entry: Optional[CacheEntry]      # populated only if hit=True

@dataclass
class QueryResult:
    response: str                    # final response returned to caller
    hit: bool                        # True if served from cache
    similarity: float                # cosine similarity of the match (0.0 on miss)
    latency_ms: float                # total end-to-end latency
    namespace: str                   # namespace used
    entry_id: Optional[str]          # cache entry ID if hit=True
    from_fallback: bool              # True if LLM was called

@dataclass
class CacheStats:
    # Counters
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    guardrail_rejections: int = 0

    # Rates
    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    # Latency (milliseconds)
    avg_total_latency_ms: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Similarity
    avg_hit_similarity: float = 0.0
    min_hit_similarity: float = 1.0

    # Token / cost savings
    tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0

    # Store state
    total_entries: int = 0
    entries_by_namespace: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "guardrail_rejections": self.guardrail_rejections,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "avg_total_latency_ms": self.avg_total_latency_ms,
            "avg_hit_latency_ms": self.avg_hit_latency_ms,
            "avg_miss_latency_ms": self.avg_miss_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_hit_similarity": self.avg_hit_similarity,
            "min_hit_similarity": self.min_hit_similarity if self.cache_hits > 0 else 0.0,
            "tokens_saved": self.tokens_saved,
            "estimated_cost_saved_usd": self.estimated_cost_saved_usd,
            "total_entries": self.total_entries,
            "entries_by_namespace": self.entries_by_namespace
        }

    def reset(self) -> None:
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.guardrail_rejections = 0
        self.avg_total_latency_ms = 0.0
        self.avg_hit_latency_ms = 0.0
        self.avg_miss_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.p99_latency_ms = 0.0
        self.avg_hit_similarity = 0.0
        self.min_hit_similarity = 1.0
        self.tokens_saved = 0
        self.estimated_cost_saved_usd = 0.0
        self.total_entries = 0
        self.entries_by_namespace.clear()

    def __str__(self) -> str:
        return (f"CacheStats(hits={self.cache_hits}/{self.total_queries} "
                f"[{self.hit_rate*100:.1f}%], avg_latency={self.avg_total_latency_ms:.1f}ms, "
                f"entries={self.total_entries})")
