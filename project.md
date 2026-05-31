# FastCache — Project Specification

> A provider-agnostic, embeddable Python library for semantic caching of LLM responses.
> Drop-in single function call. Zero required config. Works out of the box.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Installation & Setup](#3-installation--setup)
4. [Core Concepts](#4-core-concepts)
5. [Public API](#5-public-api)
6. [Configuration Reference](#6-configuration-reference)
7. [Embedder Interface & Built-in Providers](#7-embedder-interface--built-in-providers)
8. [Vector Store Interface & Built-in Providers](#8-vector-store-interface--built-in-providers)
9. [Guardrails](#9-guardrails)
10. [Exception Hierarchy](#10-exception-hierarchy)
11. [Stats System](#11-stats-system)
12. [Cache Warming](#12-cache-warming)
13. [Dashboard](#13-dashboard)
14. [Data Models](#14-data-models)
15. [Async Support](#15-async-support)
16. [Internal Flow — Step by Step](#16-internal-flow--step-by-step)
17. [Implementation Notes & Constraints](#17-implementation-notes--constraints)

---

## 1. Project Overview

### What it is

FastCache is a Python library that adds semantic caching to any LLM pipeline via a single function call. It intercepts LLM queries, checks whether a semantically similar query has been answered before, and returns the cached response if so — skipping the LLM call entirely.

### What it is NOT

- Not a managed service or SaaS
- Not an LLM router or proxy
- Not opinionated about which LLM you use
- Not tied to any specific vector database

### Core value proposition

```python
# Before FastCache
response = my_llm_function(prompt)

# After FastCache — one line change, everything else automatic
response = cache.query(prompt, my_llm_function)
```

### Design principles

- **Zero required config** — `SemanticCache()` with no args must work if `FASTCACHE_GEMINI_API_KEY` or `GEMINI_API_KEY` is set in environment
- **Pluggable everything** — embedder, vector store, guardrails are all swappable interfaces
- **Fail loudly with helpful messages** — never swallow errors silently; every failure raises a typed exception with a human-readable message and fix suggestion
- **No magic** — no monkey-patching, no import hooks, no global state

---

## 2. Directory Structure

```
fastcache/
├── fastcache/
│   ├── __init__.py                  # Public exports
│   ├── cache.py                     # SemanticCache — main entry point
│   ├── config.py                    # CacheConfig dataclass
│   ├── models.py                    # CacheEntry, QueryResult, CacheStats data models
│   ├── exceptions.py                # Full exception hierarchy
│   ├── embedders/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseEmbedder abstract class
│   │   ├── gemini.py                # GeminiEmbedder (default)
│   │   └── openai.py                # OpenAIEmbedder
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseVectorStore abstract class
│   │   ├── memory.py                # InMemoryStore (default)
│   │   └── redis_store.py           # RedisStore
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseGuardrail abstract class
│   │   └── builtin.py               # BuiltinGuardrail (length, injection)
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                   # Streamlit dashboard
├── tests/
│   ├── test_cache.py
│   ├── test_embedders.py
│   ├── test_stores.py
│   └── test_guardrails.py
├── examples/
│   ├── basic_usage.py
│   ├── with_redis.py
│   ├── custom_embedder.py
│   ├── multitenant.py
│   └── cache_warming.py
├── pyproject.toml
├── README.md
└── SPEC.md                          # this file
```

---

## 3. Installation & Setup

### Dependencies

**Core (required):**
```
numpy>=1.26.0
```

**Optional extras:**
```
# Dashboard
streamlit>=1.35.0

# Redis store
redis>=5.0.0

# OpenAI embedder
openai>=1.30.0

# sentence-transformers embedder (fully offline)
sentence-transformers>=3.0.0
```

### pyproject.toml extras definition

```toml
[project.optional-dependencies]
redis = ["redis>=5.0.0"]
openai = ["openai>=1.30.0"]
dashboard = ["streamlit>=1.35.0"]
transformers = ["sentence-transformers>=3.0.0"]
all = ["redis>=5.0.0", "openai>=1.30.0", "streamlit>=1.35.0", "sentence-transformers>=3.0.0"]
```

### Environment variables

| Variable | Description |
|---|---|
| `FASTCACHE_GEMINI_API_KEY` | Gemini API key (checked first) |
| `GEMINI_API_KEY` | Gemini API key (fallback) |
| `FASTCACHE_OPENAI_API_KEY` | OpenAI API key (checked first) |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |
| `FASTCACHE_REDIS_URL` | Redis connection URL, default `redis://localhost:6379` |
| `FASTCACHE_THRESHOLD` | Default similarity threshold, overrides 0.92 default |
| `FASTCACHE_TTL` | Default TTL in seconds, 0 = no expiry |
| `FASTCACHE_LOG_LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR |

---

## 4. Core Concepts

### Semantic similarity

Two queries are considered semantically equivalent if their embedding vectors have a cosine similarity ≥ threshold. Cosine similarity ranges from -1 to 1; in practice for text queries it ranges from 0 to 1. The default threshold of 0.92 means queries must be 92% semantically similar to produce a cache hit.

Cosine similarity formula:
```
similarity = dot(A, B) / (norm(A) * norm(B))
```

### Cache entry

Each entry in the cache stores:
- The original query text
- The embedding vector of the query
- The response text (returned by the fallback function)
- Timestamp of creation
- TTL (seconds, 0 = never expires)
- Hit counter
- Namespace

### Namespace

A string prefix that partitions the cache. Entries in different namespaces never match each other, even if their vectors are above threshold. Used for multi-tenant isolation. Default namespace is `"default"`.

### Fallback function

Any Python callable that accepts a single string (the prompt) and returns a string (the LLM response). FastCache is completely agnostic about what this function does internally.

```python
# All of these are valid fallback functions
def my_fallback(prompt: str) -> str:
    return openai_client.chat(prompt)

my_fallback = lambda p: gemini.generate(p)

# Async fallback is also supported (see Section 15)
async def my_async_fallback(prompt: str) -> str:
    return await async_llm_client.chat(prompt)
```

---

## 5. Public API

### `SemanticCache` class

```python
from fastcache import SemanticCache

cache = SemanticCache(
    embedder=None,           # BaseEmbedder instance; default: GeminiEmbedder()
    store=None,              # BaseVectorStore instance; default: InMemoryStore()
    config=None,             # CacheConfig instance; default: CacheConfig()
    guardrails=None,         # list[BaseGuardrail]; default: [BuiltinGuardrail()]
    on_hit=None,             # Callable[[str, str, float, dict], bool] | None
    dashboard=False,         # bool; if True, enables dashboard (call cache.serve_dashboard())
)
```

---

### `cache.query()`

**Primary method. This is the single entry point developers use.**

```python
response: str = cache.query(
    prompt,                  # str — the user query / LLM prompt (required)
    fallback,                # Callable[[str], str] — called on cache miss (required)
    threshold=None,          # float | None — overrides config.threshold for this call only
    namespace=None,          # str | None — overrides config.default_namespace for this call only
    ttl=None,                # int | None — overrides config.ttl for this call only
    skip_cache=False,        # bool — if True, always call fallback and store result
    no_store=False,          # bool — if True, call fallback but do NOT store result
)
```

**Behaviour:**
1. Run pre-query guardrails on `prompt`. If any guardrail rejects, raise `GuardrailError`.
2. Embed `prompt` → vector via embedder.
3. Search vector store in `namespace` for nearest neighbour above `threshold`.
4. **Cache HIT:** call `on_hit(prompt, cached_response, similarity, metadata)` if provided. If `on_hit` returns `False`, treat as miss. Otherwise return cached response.
5. **Cache MISS:** call `fallback(prompt)` → response. Store `(vector, response)` in store under `namespace` with `ttl`. Return response.
6. Update stats regardless of hit/miss.

**Returns:** `str` — the response (from cache or from fallback)

**Raises:** `GuardrailError`, `EmbedderError`, `StoreError`, `FallbackError`, `ConfigurationError`

---

### `cache.aquery()`

Async version of `query()`. Identical signature and behaviour, async-native I/O.

```python
response: str = await cache.aquery(prompt, fallback, ...)
```

The `fallback` argument may be either a sync or async callable. FastCache detects this and handles both.

---

### `cache.warm()`

Pre-populate the cache with known prompt-response pairs.

```python
cache.warm(
    entries,                 # list[tuple[str, str]] — list of (prompt, response) pairs
    namespace=None,          # str | None — namespace to warm into
    ttl=None,                # int | None — TTL for warmed entries
)
```

Also available as async: `await cache.awarm(entries, ...)`

---

### `cache.warm_from_csv()`

Convenience wrapper around `warm()` that reads from a CSV file.

```python
cache.warm_from_csv(
    path,                    # str | Path — path to CSV file
    prompt_col="prompt",     # str — column name for prompts
    response_col="response", # str — column name for responses
    namespace=None,
    ttl=None,
)
```

CSV format (default column names):
```csv
prompt,response
"What is Python?","Python is a high-level programming language..."
"What is FastAPI?","FastAPI is a modern web framework..."
```

---

### `cache.invalidate()`

Remove entries from the cache.

```python
# Remove all entries in a namespace
cache.invalidate(namespace="user:sahil")

# Remove entries matching a specific prompt (exact embedding match)
cache.invalidate(prompt="What is Python?", namespace="default")

# Clear entire cache
cache.invalidate(all=True)
```

---

### `cache.stats`

Property returning a `CacheStats` object. See Section 11.

```python
stats = cache.stats
print(stats.hit_rate)        # float
print(stats.as_dict())       # dict — for serialization
```

---

### `cache.serve_dashboard()`

Start the Streamlit monitoring dashboard. Blocks if `background=False`.

```python
cache.serve_dashboard(
    port=8501,               # int
    background=True,         # bool — if True, run in background thread (non-blocking)
)
```

Only available if `dashboard=True` was passed to the constructor. Raises `ConfigurationError` otherwise.

---

## 6. Configuration Reference

### `CacheConfig` dataclass

```python
from fastcache import CacheConfig

config = CacheConfig(
    threshold=0.92,              # float — cosine similarity threshold for cache hits (0.0–1.0)
    ttl=0,                       # int — TTL in seconds; 0 = entries never expire
    default_namespace="default", # str — namespace used when none is specified per-call
    max_cache_size=10_000,       # int — max entries in memory store (0 = unlimited)
    exact_match_first=True,      # bool — check exact string hash before embedding (free fast path)
    log_level="WARNING",         # str — DEBUG | INFO | WARNING | ERROR
)
```

### Config resolution order (highest to lowest priority)

```
per-call argument (e.g. threshold=0.85 in cache.query())
  → CacheConfig field (e.g. config.threshold)
    → Environment variable (e.g. FASTCACHE_THRESHOLD)
      → Built-in default
```

---

## 7. Embedder Interface & Built-in Providers

### `BaseEmbedder` abstract class

All embedders must implement this interface:

```python
# fastcache/embedders/base.py

from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a float32 numpy vector.
        Must return a 1D numpy array of dtype float32.
        Raise EmbedderError on any failure.
        """
        ...

    async def aembed(self, text: str) -> np.ndarray:
        """
        Async embed. Default implementation wraps embed() in asyncio.to_thread.
        Override for native async (e.g. aiohttp-based clients).
        """
        import asyncio
        return await asyncio.to_thread(self.embed, text)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension. Used for store validation."""
        raise NotImplementedError
```

### `GeminiEmbedder` (default)

```python
from fastcache.embedders import GeminiEmbedder

embedder = GeminiEmbedder(
    api_key=None,            # str | None — if None, reads FASTCACHE_GEMINI_API_KEY then GEMINI_API_KEY
    model="text-embedding-004",  # str — embedding model name
)
```

**Implementation details:**
- Uses `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent` REST endpoint
- Uses Python stdlib `urllib` only — no google-generativeai package required
- `dimension` = 768
- Raises `ConfigurationError` if no API key found with message:
  ```
  GeminiEmbedder requires an API key. Set GEMINI_API_KEY environment variable
  or pass api_key= to GeminiEmbedder(). Get a free key at https://aistudio.google.com/apikey
  ```
- Raises `EmbedderError` wrapping the HTTP error on API failure

### `OpenAIEmbedder`

```python
from fastcache.embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key=None,            # str | None — reads FASTCACHE_OPENAI_API_KEY then OPENAI_API_KEY
    model="text-embedding-3-small",  # str
)
```

- `dimension` = 1536
- Requires `pip install fastcache[openai]`
- Raises `ImportError` with install instructions if openai package not installed

### Custom embedder example

```python
from fastcache.embedders import BaseEmbedder
import numpy as np

class MyEmbedder(BaseEmbedder):
    def embed(self, text: str) -> np.ndarray:
        # your embedding logic
        return np.array([...], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return 512

cache = SemanticCache(embedder=MyEmbedder())
```

---

## 8. Vector Store Interface & Built-in Providers

### `BaseVectorStore` abstract class

```python
# fastcache/stores/base.py

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from fastcache.models import CacheEntry, LookupResult

class BaseVectorStore(ABC):

    @abstractmethod
    def store(
        self,
        vector: np.ndarray,
        query: str,
        response: str,
        namespace: str,
        ttl: int,
    ) -> CacheEntry:
        """Store a new cache entry. Returns the created CacheEntry."""
        ...

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        namespace: str,
        threshold: float,
    ) -> LookupResult:
        """
        Search for the most similar vector in the given namespace.
        Returns LookupResult with hit=True and entry populated if similarity >= threshold.
        Returns LookupResult with hit=False if no match found.
        """
        ...

    @abstractmethod
    def delete(self, namespace: str, entry_id: Optional[str] = None) -> int:
        """
        Delete entries. If entry_id is None, delete all entries in namespace.
        Returns number of entries deleted.
        """
        ...

    @abstractmethod
    def size(self, namespace: Optional[str] = None) -> int:
        """Return total number of entries. If namespace given, count only that namespace."""
        ...

    # Async methods — default to thread-wrapped sync versions
    async def astore(self, *args, **kwargs) -> CacheEntry:
        import asyncio
        return await asyncio.to_thread(self.store, *args, **kwargs)

    async def asearch(self, *args, **kwargs) -> LookupResult:
        import asyncio
        return await asyncio.to_thread(self.search, *args, **kwargs)
```

### `InMemoryStore` (default)

```python
from fastcache.stores import InMemoryStore

store = InMemoryStore(
    max_size=10_000,         # int — max entries (0 = unlimited); evicts LRU when full
)
```

**Implementation details:**
- Stores entries as a Python list of `CacheEntry` objects
- On `search()`: builds `numpy` matrix from all vectors in namespace, computes batch cosine similarity, returns best match above threshold
- TTL enforced on read: expired entries are treated as misses and lazily removed
- Thread-safe via `threading.RLock`
- When `max_size` is exceeded, evict the least recently used entry (lowest `last_accessed` timestamp)

### `RedisStore`

```python
from fastcache.stores import RedisStore

store = RedisStore(
    url="redis://localhost:6379",  # str — Redis URL; reads FASTCACHE_REDIS_URL if not set
    key_prefix="fastcache",        # str — prefix for all Redis keys
    use_vector_sets=True,          # bool — use Redis 8 native vector sets if available;
                                   #        falls back to manual cosine search otherwise
)
```

**Implementation details:**
- Requires `pip install fastcache[redis]`
- Each cache entry stored as Redis hash: `{key_prefix}:{namespace}:entry:{id}`
- Vector stored as bytes (numpy `.tobytes()`) in the hash
- TTL set natively via Redis `EXPIRE`
- On `search()`: retrieves all vectors in namespace, computes cosine similarity in numpy (Redis vector sets used if available on Redis 8+)
- Raises `StoreError` with connection details if Redis unreachable
- Raises `ImportError` with install instructions if `redis` package not installed

### Custom store example

```python
from fastcache.stores import BaseVectorStore
from fastcache.models import CacheEntry, LookupResult

class MyVectorStore(BaseVectorStore):
    def store(self, vector, query, response, namespace, ttl) -> CacheEntry:
        ...

    def search(self, vector, namespace, threshold) -> LookupResult:
        ...

    def delete(self, namespace, entry_id=None) -> int:
        ...

    def size(self, namespace=None) -> int:
        ...

cache = SemanticCache(store=MyVectorStore())
```

---

## 9. Guardrails

Guardrails are validation hooks that run before the cache lookup. They can reject a query entirely or allow it to proceed.

### `BaseGuardrail` abstract class

```python
# fastcache/guardrails/base.py

from abc import ABC, abstractmethod

class BaseGuardrail(ABC):

    @abstractmethod
    def validate(self, prompt: str) -> tuple[bool, str]:
        """
        Validate the prompt.
        Returns (True, "") if the prompt is acceptable.
        Returns (False, reason) if the prompt should be rejected.
        Raise GuardrailError for hard failures.
        """
        ...
```

### `BuiltinGuardrail` (default, always active)

Enabled by default. Cannot be disabled but can be configured.

```python
from fastcache.guardrails import BuiltinGuardrail

guardrail = BuiltinGuardrail(
    min_length=3,             # int — reject prompts shorter than N characters
    max_length=32_000,        # int — reject prompts longer than N characters
    block_injections=True,    # bool — detect common prompt injection patterns
)
```

**Built-in checks:**

| Check | Default | Description |
|---|---|---|
| `min_length` | 3 chars | Rejects queries too short to be meaningful |
| `max_length` | 32,000 chars | Rejects queries too large to embed reliably |
| `block_injections` | True | Detects patterns like `ignore previous instructions`, `system:`, `<|im_start|>` |

### `on_hit` callback — post-cache guardrail

Called after a cache hit is found, before returning the response. Gives the developer full control to reject a hit at runtime.

```python
def my_on_hit(
    query: str,              # the current query
    cached_response: str,    # the cached response about to be returned
    similarity: float,       # cosine similarity score (0.0–1.0)
    metadata: dict,          # entry metadata: namespace, timestamp, hit_count, ttl
) -> bool:
    """
    Return True  → accept the cache hit, return cached_response to caller
    Return False → reject the hit, treat as miss, call LLM fallback
    """
    # Example: reject cached responses older than 1 hour for time-sensitive topics
    if metadata["age_seconds"] > 3600 and "price" in query.lower():
        return False
    return True

cache = SemanticCache(on_hit=my_on_hit)
```

### Custom guardrail example

```python
from fastcache.guardrails import BaseGuardrail

class TopicGuardrail(BaseGuardrail):
    def __init__(self, allowed_topics: list[str]):
        self.allowed_topics = allowed_topics

    def validate(self, prompt: str) -> tuple[bool, str]:
        prompt_lower = prompt.lower()
        if not any(topic in prompt_lower for topic in self.allowed_topics):
            return False, f"Query not related to allowed topics: {self.allowed_topics}"
        return True, ""

cache = SemanticCache(
    guardrails=[TopicGuardrail(allowed_topics=["python", "fastapi", "redis"])]
)
```

**Guardrail execution order:**
1. `BuiltinGuardrail` always runs first
2. Additional guardrails in the list run in order
3. First rejection stops execution and raises `GuardrailError`
4. `on_hit` runs last, after similarity search

---

## 10. Exception Hierarchy

```
FastCacheError                          # base for all library exceptions
├── ConfigurationError                  # bad or missing config / API keys
├── EmbedderError                       # embedding failed
│   └── EmbedderAuthError               # bad API key
├── StoreError                          # vector store operation failed
│   └── StoreConnectionError            # cannot connect to Redis
├── GuardrailError                      # pre-query validation rejected the prompt
├── FallbackError                       # fallback function raised an exception
└── CacheWarmingError                   # warm() or warm_from_csv() failed
```

### Exception design rules

Every exception must include:
- A human-readable `message` describing what went wrong
- A `suggestion` string explaining how to fix it
- The original exception as `__cause__` where applicable

```python
# Example — how exceptions should look in practice
raise ConfigurationError(
    message="GeminiEmbedder requires an API key but none was found.",
    suggestion=(
        "Set the GEMINI_API_KEY environment variable, or pass api_key= to "
        "GeminiEmbedder(). Get a free key at https://aistudio.google.com/apikey"
    )
)

raise StoreConnectionError(
    message="Cannot connect to Redis at redis://localhost:6379.",
    suggestion=(
        "Ensure Redis is running: `docker run -p 6379:6379 redis`. "
        "Or use the default InMemoryStore by not passing a store= argument."
    ),
    cause=original_exception,
)
```

---

## 11. Stats System

### `CacheStats` dataclass

Returned by `cache.stats` property. Always reflects the current state.

```python
@dataclass
class CacheStats:
    # Counters
    total_queries: int           # total calls to query() / aquery()
    cache_hits: int              # queries served from cache
    cache_misses: int            # queries that called fallback
    guardrail_rejections: int    # queries rejected by guardrails

    # Rates
    hit_rate: float              # cache_hits / total_queries (0.0 if no queries)
    miss_rate: float             # 1.0 - hit_rate

    # Latency (milliseconds)
    avg_total_latency_ms: float  # average end-to-end latency
    avg_hit_latency_ms: float    # average latency on cache hits
    avg_miss_latency_ms: float   # average latency on cache misses (includes LLM)
    p95_latency_ms: float        # 95th percentile end-to-end latency
    p99_latency_ms: float        # 99th percentile end-to-end latency

    # Similarity
    avg_hit_similarity: float    # average cosine similarity on cache hits
    min_hit_similarity: float    # lowest similarity that still produced a hit

    # Token / cost savings (populated only when fallback returns metadata)
    tokens_saved: int            # tokens avoided by cache hits
    estimated_cost_saved_usd: float

    # Store state
    total_entries: int           # current number of entries across all namespaces
    entries_by_namespace: dict[str, int]  # breakdown per namespace

    # Methods
    def as_dict(self) -> dict: ...           # JSON-serializable dict
    def reset(self) -> None: ...             # reset all counters to zero
    def __str__(self) -> str: ...            # human-readable summary
```

---

## 12. Cache Warming

### `cache.warm()`

Pre-populate the cache before any user traffic arrives.

```python
cache.warm(
    entries=[
        ("What is your refund policy?", "Our refund policy allows returns within 30 days..."),
        ("How do I reset my password?", "To reset your password, click 'Forgot Password'..."),
    ],
    namespace="support-bot",
    ttl=86400,               # 24 hours
)
```

**Behaviour:**
- Embeds each prompt, stores `(vector, response)` in the store
- If a prompt is already in the cache above threshold, skip it (no duplicate)
- Raises `CacheWarmingError` if embedding fails for any entry, with details on which entry failed
- Progress is logged at INFO level: `Warmed 47/50 entries (3 skipped as duplicates)`

### `cache.warm_from_csv()`

```python
cache.warm_from_csv(
    path="./faq.csv",
    prompt_col="question",
    response_col="answer",
    namespace="faq",
)
```

CSV requirements:
- UTF-8 encoded
- Must contain `prompt_col` and `response_col` columns (configurable)
- Empty rows are skipped
- Rows where either column is empty are skipped with a WARNING log

---

## 13. Dashboard

### Overview

An optional Streamlit-based monitoring dashboard. Shows real-time stats, cache entry explorer, and similarity heatmap.

### Activation

```python
# Must pass dashboard=True to constructor to enable
cache = SemanticCache(dashboard=True)

# Then call serve_dashboard() explicitly — never auto-starts in constructor
cache.serve_dashboard(port=8501, background=True)
```

Calling `serve_dashboard()` without `dashboard=True` raises:
```
ConfigurationError: Dashboard is not enabled. Pass dashboard=True to SemanticCache().
```

Calling `serve_dashboard()` without `streamlit` installed raises:
```
ImportError: Dashboard requires streamlit. Install it with: pip install fastcache[dashboard]
```

### Dashboard pages / sections

**1. Overview (always visible top bar)**
- Total queries, cache hit rate %, avg latency, tokens saved, entries in cache

**2. Live Query Feed**
- Last 50 queries in real-time
- Hit/miss badge, similarity score, latency, namespace

**3. Stats Charts**
- Hit rate over time (line chart)
- Latency distribution: cache hits vs LLM calls (bar chart)
- Queries per namespace (pie chart)

**4. Cache Explorer**
- Table of all cached entries with: ID, namespace, query text, hit count, age, TTL remaining
- Filter by namespace
- Search by query text
- Delete individual entries or entire namespace

**5. Similarity Heatmap**
- Cosine similarity matrix for all entries (shown when ≤ 25 entries)
- Color-coded: green = above threshold (would hit), blue = close, dark = distant

---

## 14. Data Models

```python
# fastcache/models.py

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time
import uuid

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
```

---

## 15. Async Support

Both `cache.query()` (sync) and `cache.aquery()` (async) are first-class methods.

### Sync usage

```python
cache = SemanticCache()
response = cache.query(prompt, my_llm_fn)
```

### Async usage

```python
cache = SemanticCache()

async def main():
    response = await cache.aquery(prompt, my_async_llm_fn)
```

### Mixed (sync fallback in async context)

FastCache detects whether `fallback` is a coroutine function and handles both:

```python
# This works — FastCache wraps sync fallback in asyncio.to_thread
async def main():
    response = await cache.aquery(prompt, sync_llm_fn)
```

### Embedder and store async

- Both `BaseEmbedder` and `BaseVectorStore` have async methods
- Default implementations wrap sync methods in `asyncio.to_thread`
- Override `aembed()` / `asearch()` / `astore()` in custom implementations for native async

---

## 16. Internal Flow — Step by Step

### Sync `cache.query(prompt, fallback, **kwargs)`

```
1. Resolve config
   ├── threshold = kwargs.threshold ?? config.threshold
   ├── namespace = kwargs.namespace ?? config.default_namespace
   └── ttl       = kwargs.ttl       ?? config.ttl

2. Run guardrails
   ├── For each guardrail in [BuiltinGuardrail] + config.guardrails:
   │   └── valid, reason = guardrail.validate(prompt)
   │       └── if not valid → raise GuardrailError(reason)
   └── (guardrail_rejections stat incremented on rejection)

3. Exact match fast path (if config.exact_match_first=True)
   ├── key = sha256(namespace + ":" + prompt.strip().lower())
   ├── lookup key in exact_match_index dict
   └── if found and not expired → skip embedding, return cached response (counts as hit)

4. Embed
   ├── vector = embedder.embed(prompt)
   └── on failure → raise EmbedderError

5. Search vector store
   ├── result = store.search(vector, namespace, threshold)
   └── on failure → raise StoreError

6. Branch on result.hit
   ├── HIT:
   │   ├── if on_hit is set:
   │   │   ├── metadata = {namespace, age_seconds, hit_count, ttl_remaining, similarity}
   │   │   ├── accepted = on_hit(prompt, result.entry.response, result.similarity, metadata)
   │   │   └── if not accepted → go to MISS branch
   │   ├── increment result.entry.hit_count
   │   ├── update result.entry.last_accessed
   │   └── return result.entry.response
   │
   └── MISS:
       ├── try: response = fallback(prompt)
       │   └── on exception → raise FallbackError(cause=original)
       ├── store.store(vector, prompt, response, namespace, ttl)
       └── return response

7. Update stats (always, in finally block)
   ├── increment total_queries
   ├── increment cache_hits or cache_misses
   ├── record latency_ms
   └── record similarity
```

---

## 17. Implementation Notes & Constraints

### Thread safety
- `InMemoryStore` must use `threading.RLock` for all read/write operations
- `CacheStats` counters must use `threading.Lock` or `collections.Counter` for atomic increments

### Cosine similarity implementation
- Always normalize vectors before storing to enable faster dot-product-only similarity on search
- Batch computation: stack all namespace vectors into a matrix, compute all similarities in one `np.dot()` call
- Never loop over entries one by one for similarity computation

### TTL enforcement
- TTL is enforced lazily on read (not via background thread)
- `InMemoryStore.search()` skips expired entries and treats them as misses
- Periodic cleanup is optional and only done if `max_size` is set and the store is nearly full

### Exact match fast path
- Store a `dict[str, CacheEntry]` indexed by `sha256(namespace + ":" + normalized_prompt)` alongside the vector store
- Check this dict before embedding — saves embedding API call on exact duplicate queries
- Enabled by default via `CacheConfig.exact_match_first=True`

### Dashboard isolation
- Dashboard must never import from `fastcache` in a way that creates a new cache instance
- Stats are passed by reference; the dashboard reads from the live `CacheStats` object
- Dashboard runs in a `threading.Thread` (daemon=True) so it doesn't block the main program

### No global state
- No module-level cache instances
- No `fastcache.configure()` global config function
- All state is on the `SemanticCache` instance

### Logging
- Use Python's `logging` module with logger name `fastcache`
- Never `print()` anything — all output through the logger
- Log levels:
  - DEBUG: embedding vectors, similarity scores, store operations
  - INFO: cache hits/misses, warming progress
  - WARNING: TTL expiry, skipped warm entries, on_hit rejections
  - ERROR: embedder failures, store failures (before raising exceptions)

### Package exports (`fastcache/__init__.py`)

```python
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
```

---

## Appendix A — Minimal working example

```python
import os
from fastcache import SemanticCache

# GEMINI_API_KEY set in environment
cache = SemanticCache()

def my_llm(prompt: str) -> str:
    # your actual LLM call here
    return f"Answer to: {prompt}"

# First call — cache miss, calls my_llm
response = cache.query("What is Python?", my_llm)

# Second call — cache hit (above 0.92 threshold), my_llm NOT called
response = cache.query("Can you explain Python to me?", my_llm)

print(cache.stats)
```

## Appendix B — Full configuration example

```python
from fastcache import SemanticCache, CacheConfig
from fastcache.embedders import GeminiEmbedder
from fastcache.stores import RedisStore
from fastcache.guardrails import BuiltinGuardrail

def on_hit(query, response, similarity, metadata):
    # reject stale hits for price-related queries
    if metadata["age_seconds"] > 3600 and "price" in query.lower():
        return False
    return True

cache = SemanticCache(
    embedder=GeminiEmbedder(api_key="AIza..."),
    store=RedisStore(url="redis://localhost:6379"),
    config=CacheConfig(
        threshold=0.92,
        ttl=3600,
        default_namespace="myapp",
        max_cache_size=50_000,
        exact_match_first=True,
    ),
    guardrails=[
        BuiltinGuardrail(min_length=5, max_length=10_000, block_injections=True),
    ],
    on_hit=on_hit,
    dashboard=True,
)

cache.warm_from_csv("./faq.csv", namespace="faq")
cache.serve_dashboard(port=8501, background=True)

# Per-call overrides
response = cache.query(
    "What is the price of product X?",
    my_llm,
    threshold=0.98,          # stricter for price queries
    namespace="pricing",
    ttl=300,                 # 5 min TTL for pricing data
)
```

## Appendix C — Redis setup (quickstart)

```bash
# Docker (easiest)
docker run -d -p 6379:6379 --name fastcache-redis redis:8

# Verify
redis-cli ping  # should return PONG
```

```python
from fastcache.stores import RedisStore
store = RedisStore(url="redis://localhost:6379")
```