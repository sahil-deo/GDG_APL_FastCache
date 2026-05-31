from fastcache.cache import SemanticCache
from fastcache.config import CacheConfig
from fastcache.models import CacheEntry, QueryResult, CacheStats, LookupResult
from fastcache.exceptions import (
    FastCacheError,
    ConfigurationError,
    EmbedderError,
    EmbedderAuthError,
    StoreError,
    StoreConnectionError,
    GuardrailError,
    FallbackError,
    CacheWarmingError,
)
from fastcache.embedders import BaseEmbedder, GeminiEmbedder, OpenAIEmbedder
from fastcache.stores import BaseVectorStore, InMemoryStore, RedisStore
from fastcache.guardrails import BaseGuardrail, BuiltinGuardrail

__version__ = "0.1.0"
__all__ = [
    "SemanticCache", "CacheConfig",
    "CacheEntry", "QueryResult", "CacheStats", "LookupResult",
    "FastCacheError", "ConfigurationError", "EmbedderError", "EmbedderAuthError",
    "StoreError", "StoreConnectionError", "GuardrailError", "FallbackError", "CacheWarmingError",
    "BaseEmbedder", "GeminiEmbedder", "OpenAIEmbedder",
    "BaseVectorStore", "InMemoryStore", "RedisStore",
    "BaseGuardrail", "BuiltinGuardrail",
]
