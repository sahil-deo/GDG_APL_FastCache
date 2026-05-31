"""
dashboard_demo.py
-----------------
Seeds FastCache with synthetic data and launches the built-in
administrative dashboard on http://localhost:8501.

Run:
    uv run python examples/dashboard_demo.py

No API key required — uses a local random embedder.
"""

import sys
import os
import time
import numpy as np
import subprocess


# Ensure the library is importable when run from the fastcache/ project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastcache import SemanticCache
from fastcache.config import CacheConfig
from fastcache.embedders.gemini import GeminiEmbedder
from fastcache.stores.memory import InMemoryStore


# ─────────────────────────────────────────────────────────────
# Synthetic Q&A pairs to seed the cache
# ─────────────────────────────────────────────────────────────
SEED_DATA = [
    ("What is Python?",           "Python is a high-level, interpreted programming language known for its simplicity.", "general"),
    ("Explain Python briefly.",   "Python is a high-level, interpreted programming language known for its simplicity.", "general"),  # hit
    ("What is FastAPI?",          "FastAPI is a modern, high-performance web framework for building APIs with Python.", "general"),
    ("How does FastAPI work?",    "FastAPI is a modern, high-performance web framework for building APIs with Python.", "general"),  # hit
    ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data.", "ml"),
    ("Explain ML in short.",      "Machine learning is a subset of AI that enables systems to learn from data.", "ml"),             # hit
    ("What is a neural network?", "A neural network is a series of algorithms that mimic the human brain.", "ml"),
    ("What is Redis?",            "Redis is an open-source, in-memory data structure store used as a database and cache.", "infra"),
    ("How does Redis work?",      "Redis is an open-source, in-memory data structure store used as a database and cache.", "infra"), # hit
    ("What is Docker?",           "Docker is a platform for developing, shipping, and running applications in containers.", "infra"),
    ("Explain containers.",       "Containers are lightweight, portable units that package code and its dependencies.", "infra"),
    ("What is Kubernetes?",       "Kubernetes is an open-source container orchestration system for automating deployment.", "infra"),
]


def seed_cache(cache: SemanticCache):
    """Directly populate the store with synthetic entries and fake stats."""
    store: InMemoryStore = cache.store
    embedder: GeminiEmbedder = cache.embedder

    print("Seeding cache with synthetic data...")
    for query, response, namespace in SEED_DATA:
        vector = embedder.embed(query)
        entry = store.store(vector, query, response, namespace, ttl=0)
        print(f"  ✓  [{namespace}] {query[:55]}")

    # Simulate stats: 8 hits, 4 misses over 12 queries
    with cache._stats_lock:
        cache._stats.total_queries        = 12
        cache._stats.cache_hits           = 8
        cache._stats.cache_misses         = 4
        cache._stats.avg_total_latency_ms = 320.0
        cache._stats.avg_hit_latency_ms   = 45.0
        cache._stats.avg_miss_latency_ms  = 890.0
        cache._stats.tokens_saved         = 8 * 220
        cache._stats.avg_hit_similarity   = 0.91
        cache._stats.p95_latency_ms       = 780.0

    print(f"\nCache seeded: {store.size()} entries across "
          f"{len(store._entries)} namespaces.")
    print(f"Simulated stats: 8 hits / 4 misses (66.7% hit rate)\n")


def gemini_llm(prompt: str) -> str:
    import json, urllib.request
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("FASTCACHE_GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    return result["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    config = CacheConfig(threshold=0.85)
    cache  = SemanticCache(
        embedder=GeminiEmbedder(model="gemini-embedding-2"),
        store=InMemoryStore(),
        config=config,
        dashboard=True,
    )

    seed_cache(cache)

    print("Launching admin dashboard on http://localhost:8501 ...")
    print("Press Ctrl+C to stop.\n")

    import json, tempfile, os
    stats = cache.stats.as_dict()
    entries = []
    if hasattr(cache.store, "_entries"):
        for ns, ns_entries in cache.store._entries.items():
            for e in ns_entries:
                entries.append({
                    "id": e.id,
                    "namespace": e.namespace,
                    "query": e.query,
                    "hit_count": e.hit_count,
                    "age_seconds": e.age_seconds,
                    "ttl": e.ttl,
                    "ttl_remaining": e.ttl_remaining,
                })

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump({"stats": stats, "entries": entries}, tmp)
    tmp.close()
    os.environ["FASTCACHE_DEMO_CACHE"] = tmp.name

    import fastcache.dashboard.app as dash_app
    app_path = dash_app.__file__
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501"])

