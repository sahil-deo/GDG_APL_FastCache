import os
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class CacheConfig:
    threshold: float = field(
        default_factory=lambda: float(os.environ.get("FASTCACHE_THRESHOLD", "0.85"))
    )
    ttl: int = field(
        default_factory=lambda: int(os.environ.get("FASTCACHE_TTL", "0"))
    )
    default_namespace: str = "default"
    max_cache_size: int = 10_000
    exact_match_first: bool = True
    log_level: str = field(
        default_factory=lambda: os.environ.get("FASTCACHE_LOG_LEVEL", "WARNING").upper()
    )
