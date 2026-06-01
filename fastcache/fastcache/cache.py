import time
import hashlib
import csv
import logging
import inspect
import asyncio
import threading
from typing import Optional, Callable, Any, Coroutine, Union
from pathlib import Path

from fastcache.config import CacheConfig
from fastcache.models import CacheStats, QueryResult
from fastcache.exceptions import (
    ConfigurationError, EmbedderError, StoreError, 
    GuardrailError, FallbackError, CacheWarmingError
)
from fastcache.embedders.base import BaseEmbedder
from fastcache.embedders.gemini import GeminiEmbedder
from fastcache.stores.base import BaseVectorStore
from fastcache.stores.memory import InMemoryStore
from fastcache.guardrails.base import BaseGuardrail
from fastcache.guardrails.builtin import BuiltinGuardrail

logger = logging.getLogger("fastcache")

class SemanticCache:
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        store: Optional[BaseVectorStore] = None,
        config: Optional[CacheConfig] = None,
        guardrails: Optional[list[BaseGuardrail]] = None,
        on_hit: Optional[Callable[[str, str, float, dict], bool]] = None,
        dashboard: bool = False,
    ):
        self.config = config or CacheConfig()
        
        # Setup logging based on config
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.WARNING))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        self.embedder = embedder if embedder is not None else GeminiEmbedder()
        self.store = store if store is not None else InMemoryStore(max_size=self.config.max_cache_size)
        
        # Ensure guardrails list includes BuiltinGuardrail first
        self.guardrails = [BuiltinGuardrail()]
        if guardrails:
            self.guardrails.extend(guardrails)
            
        self.on_hit = on_hit
        self._dashboard_enabled = dashboard
        
        # Exact match index: sha256(namespace:prompt) -> response
        self._exact_match_index: dict[str, dict] = {}
        self._exact_match_lock = threading.RLock()
        
        self._stats = CacheStats()
        self._stats_lock = threading.Lock()

    @property
    def stats(self) -> CacheStats:
        with self._stats_lock:
            # Sync total entries by namespace with the store state
            total = 0
            ns_dict = {}
            
            # Note: For RedisStore, this might be expensive if called frequently.
            # But we follow the spec.
            if isinstance(self.store, InMemoryStore):
                with self.store._lock:
                    for ns, entries in self.store._entries.items():
                        c = len([e for e in entries if not e.is_expired])
                        ns_dict[ns] = c
                        total += c
            else:
                # Basic sync for other stores like Redis (would just count keys roughly)
                pass # Ideally we would query the store for size per namespace if we track namespaces
                
            self._stats.total_entries = self.store.size()
            self._stats.entries_by_namespace = ns_dict
            return self._stats

    def _resolve_config(self, kwargs_dict: dict, key: str) -> Any:
        if kwargs_dict.get(key) is not None:
            return kwargs_dict[key]
        return getattr(self.config, key if key != "namespace" else "default_namespace")

    def _get_exact_match_key(self, namespace: str, prompt: str) -> str:
        s = f"{namespace}:{prompt.strip().lower()}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _run_guardrails(self, prompt: str):
        for guardrail in self.guardrails:
            valid, reason = guardrail.validate(prompt)
            if not valid:
                with self._stats_lock:
                    self._stats.guardrail_rejections += 1
                raise GuardrailError(f"Prompt rejected by guardrail: {reason}")

    def query(
        self,
        prompt: str,
        fallback: Callable[[str], str],
        threshold: Optional[float] = None,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        skip_cache: bool = False,
        no_store: bool = False,
    ) -> str:
        start_time = time.time()
        
        eff_threshold = self._resolve_config({"threshold": threshold}, "threshold")
        eff_namespace = self._resolve_config({"namespace": namespace}, "namespace")
        eff_ttl = self._resolve_config({"ttl": ttl}, "ttl")

        try:
            self._run_guardrails(prompt)
            
            # Exact match fast path
            if self.config.exact_match_first and not skip_cache:
                key = self._get_exact_match_key(eff_namespace, prompt)
                with self._exact_match_lock:
                    entry_dict = self._exact_match_index.get(key)
                    if entry_dict:
                        # Check exact match TTL locally
                        created_at = entry_dict["created_at"]
                        entry_ttl = entry_dict["ttl"]
                        if entry_ttl == 0 or (time.time() - created_at) <= entry_ttl:
                            logger.info("Exact match fast path hit.")
                            self._update_stats(hit=True, similarity=1.0, latency_ms=(time.time()-start_time)*1000)
                            return entry_dict["response"]
                        else:
                            del self._exact_match_index[key]

            # Vector based search
            if not skip_cache:
                try:
                    vector = self.embedder.embed(prompt)
                except Exception as e:
                    if not isinstance(e, EmbedderError):
                        raise EmbedderError("Failed to embed prompt", cause=e)
                    raise e
                    
                try:
                    result = self.store.search(vector, eff_namespace, eff_threshold)
                except Exception as e:
                    if not isinstance(e, StoreError):
                        raise StoreError("Failed to search store", cause=e)
                    raise e
                    
                if result.hit and result.entry:
                    accepted = True
                    if self.on_hit:
                        metadata = {
                            "namespace": result.entry.namespace,
                            "age_seconds": result.entry.age_seconds,
                            "hit_count": result.entry.hit_count,
                            "ttl_remaining": result.entry.ttl_remaining,
                            "similarity": result.similarity
                        }
                        accepted = self.on_hit(prompt, result.entry.response, result.similarity, metadata)
                    
                    if accepted:
                        logger.info(f"Cache hit with similarity {result.similarity:.4f}")
                        result.entry.hit_count += 1
                        result.entry.last_accessed = time.time()
                        self._update_stats(hit=True, similarity=result.similarity, latency_ms=(time.time()-start_time)*1000)
                        return result.entry.response
                    else:
                        logger.warning("Cache hit rejected by on_hit callback.")

            # Fallback (Cache Miss or skip_cache or rejected hit)
            try:
                # Execute fallback sync
                response = fallback(prompt)
                if inspect.iscoroutine(response):
                    raise FallbackError("Sync query() called with an async fallback. Use aquery() instead.")
            except Exception as e:
                if not isinstance(e, FallbackError):
                    raise FallbackError("Fallback function raised an exception", cause=e)
                raise e

            if not no_store:
                if skip_cache:
                    vector = self.embedder.embed(prompt)
                
                try:
                    entry = self.store.store(vector, prompt, response, eff_namespace, eff_ttl)
                except Exception as e:
                    logger.error(f"Failed to store entry: {e}")
                    
                # Store exact match
                if self.config.exact_match_first:
                    key = self._get_exact_match_key(eff_namespace, prompt)
                    with self._exact_match_lock:
                        self._exact_match_index[key] = {
                            "response": response,
                            "created_at": time.time(),
                            "ttl": eff_ttl
                        }
            
            logger.info("Cache miss, returning fallback response.")
            self._update_stats(hit=False, similarity=0.0, latency_ms=(time.time()-start_time)*1000)
            return response

        except Exception:
            # Stats for failed queries? Spec doesn't strictly say, but usually don't count towards hits/misses, or count as misses.
            raise

    async def aquery(
        self,
        prompt: str,
        fallback: Callable[[str], Union[str, Coroutine[Any, Any, str]]],
        threshold: Optional[float] = None,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        skip_cache: bool = False,
        no_store: bool = False,
    ) -> str:
        start_time = time.time()
        
        eff_threshold = self._resolve_config({"threshold": threshold}, "threshold")
        eff_namespace = self._resolve_config({"namespace": namespace}, "namespace")
        eff_ttl = self._resolve_config({"ttl": ttl}, "ttl")

        try:
            self._run_guardrails(prompt)
            
            # Exact match fast path
            if self.config.exact_match_first and not skip_cache:
                key = self._get_exact_match_key(eff_namespace, prompt)
                with self._exact_match_lock:
                    entry_dict = self._exact_match_index.get(key)
                    if entry_dict:
                        created_at = entry_dict["created_at"]
                        entry_ttl = entry_dict["ttl"]
                        if entry_ttl == 0 or (time.time() - created_at) <= entry_ttl:
                            logger.info("Exact match fast path hit (async).")
                            self._update_stats(hit=True, similarity=1.0, latency_ms=(time.time()-start_time)*1000)
                            return entry_dict["response"]
                        else:
                            del self._exact_match_index[key]

            # Vector based search
            if not skip_cache:
                try:
                    vector = await self.embedder.aembed(prompt)
                except Exception as e:
                    if not isinstance(e, EmbedderError):
                        raise EmbedderError("Failed to embed prompt", cause=e)
                    raise e
                    
                try:
                    result = await self.store.asearch(vector, eff_namespace, eff_threshold)
                except Exception as e:
                    if not isinstance(e, StoreError):
                        raise StoreError("Failed to search store", cause=e)
                    raise e
                    
                if result.hit and result.entry:
                    accepted = True
                    if self.on_hit:
                        metadata = {
                            "namespace": result.entry.namespace,
                            "age_seconds": result.entry.age_seconds,
                            "hit_count": result.entry.hit_count,
                            "ttl_remaining": result.entry.ttl_remaining,
                            "similarity": result.similarity
                        }
                        accepted = self.on_hit(prompt, result.entry.response, result.similarity, metadata)
                    
                    if accepted:
                        logger.info(f"Cache hit with similarity {result.similarity:.4f} (async)")
                        result.entry.hit_count += 1
                        result.entry.last_accessed = time.time()
                        self._update_stats(hit=True, similarity=result.similarity, latency_ms=(time.time()-start_time)*1000)
                        return result.entry.response
                    else:
                        logger.warning("Cache hit rejected by on_hit callback. (async)")

            # Fallback
            try:
                res = fallback(prompt)
                if inspect.iscoroutine(res):
                    response = await res
                else:
                    response = res
            except Exception as e:
                if not isinstance(e, FallbackError):
                    raise FallbackError("Fallback function raised an exception", cause=e)
                raise e

            if not no_store:
                if skip_cache:
                    vector = await self.embedder.aembed(prompt)
                
                try:
                    entry = await self.store.astore(vector, prompt, response, eff_namespace, eff_ttl)
                except Exception as e:
                    logger.error(f"Failed to store entry (async): {e}")
                    
                if self.config.exact_match_first:
                    key = self._get_exact_match_key(eff_namespace, prompt)
                    with self._exact_match_lock:
                        self._exact_match_index[key] = {
                            "response": response,
                            "created_at": time.time(),
                            "ttl": eff_ttl
                        }
            
            logger.info("Cache miss, returning fallback response (async).")
            self._update_stats(hit=False, similarity=0.0, latency_ms=(time.time()-start_time)*1000)
            return response

        except Exception:
            raise

    def _update_stats(self, hit: bool, similarity: float, latency_ms: float):
        with self._stats_lock:
            self._stats.total_queries += 1
            n = self._stats.total_queries
            
            # Update avg total latency
            prev_avg = self._stats.avg_total_latency_ms
            self._stats.avg_total_latency_ms = prev_avg + (latency_ms - prev_avg) / n
            
            if hit:
                self._stats.cache_hits += 1
                h = self._stats.cache_hits
                prev_h_avg = self._stats.avg_hit_latency_ms
                self._stats.avg_hit_latency_ms = prev_h_avg + (latency_ms - prev_h_avg) / h
                
                prev_sim_avg = self._stats.avg_hit_similarity
                self._stats.avg_hit_similarity = prev_sim_avg + (similarity - prev_sim_avg) / h
                
                if similarity < self._stats.min_hit_similarity:
                    self._stats.min_hit_similarity = similarity
            else:
                self._stats.cache_misses += 1
                m = self._stats.cache_misses
                prev_m_avg = self._stats.avg_miss_latency_ms
                self._stats.avg_miss_latency_ms = prev_m_avg + (latency_ms - prev_m_avg) / m

    def warm(
        self,
        entries: list[tuple[str, str]],
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        eff_namespace = self._resolve_config({"namespace": namespace}, "namespace")
        eff_ttl = self._resolve_config({"ttl": ttl}, "ttl")
        
        skipped = 0
        warmed = 0
        
        for prompt, response in entries:
            try:
                # Check if it already exists above threshold
                vector = self.embedder.embed(prompt)
                result = self.store.search(vector, eff_namespace, self.config.threshold)
                if result.hit:
                    skipped += 1
                    continue
                
                self.store.store(vector, prompt, response, eff_namespace, eff_ttl)
                warmed += 1
            except Exception as e:
                raise CacheWarmingError(f"Failed to warm entry '{prompt}': {e}", cause=e)
                
        logger.info(f"Warmed {warmed}/{len(entries)} entries ({skipped} skipped as duplicates)")

    async def awarm(
        self,
        entries: list[tuple[str, str]],
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        eff_namespace = self._resolve_config({"namespace": namespace}, "namespace")
        eff_ttl = self._resolve_config({"ttl": ttl}, "ttl")
        
        skipped = 0
        warmed = 0
        
        for prompt, response in entries:
            try:
                vector = await self.embedder.aembed(prompt)
                result = await self.store.asearch(vector, eff_namespace, self.config.threshold)
                if result.hit:
                    skipped += 1
                    continue
                
                await self.store.astore(vector, prompt, response, eff_namespace, eff_ttl)
                warmed += 1
            except Exception as e:
                raise CacheWarmingError(f"Failed to warm entry '{prompt}': {e}", cause=e)
                
        logger.info(f"Warmed {warmed}/{len(entries)} entries ({skipped} skipped as duplicates)")

    def warm_from_csv(
        self,
        path: Union[str, Path],
        prompt_col: str = "prompt",
        response_col: str = "response",
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
    ):
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if prompt_col not in reader.fieldnames or response_col not in reader.fieldnames:
                    raise ConfigurationError(f"CSV must contain '{prompt_col}' and '{response_col}' columns.")
                
                for row in reader:
                    p = row.get(prompt_col, "").strip()
                    r = row.get(response_col, "").strip()
                    if not p or not r:
                        logger.warning("Skipped empty row in CSV.")
                        continue
                    entries.append((p, r))
        except Exception as e:
            raise CacheWarmingError(f"Failed to read CSV for warming: {e}", cause=e)
            
        self.warm(entries, namespace, ttl)

    def invalidate(self, prompt: Optional[str] = None, namespace: Optional[str] = None, all: bool = False):
        if all:
            if isinstance(self.store, InMemoryStore):
                with self.store._lock:
                    self.store._entries.clear()
            elif hasattr(self.store, 'client'):
                # rough flush logic for Redis keys matching prefix
                pattern = f"{self.store.key_prefix}:*"
                keys = self.store.client.keys(pattern)
                if keys:
                    self.store.client.delete(*keys)
            with self._exact_match_lock:
                self._exact_match_index.clear()
            logger.info("Cleared entire cache.")
            return

        eff_namespace = self._resolve_config({"namespace": namespace}, "namespace")
        
        if prompt:
            # Find the specific entry via embedding match? 
            # Or exact match fast path? The spec says:
            # "Remove entries matching a specific prompt (exact embedding match)"
            # It's tricky with vector stores to find exact embedding match without search.
            # Best effort: search and delete if similarity ~ 1.0
            vector = self.embedder.embed(prompt)
            result = self.store.search(vector, eff_namespace, threshold=0.99)
            if result.hit and result.entry:
                self.store.delete(eff_namespace, result.entry.id)
                logger.info(f"Invalidated exact match for prompt in {eff_namespace}")
            
            # also remove from exact_match_index
            key = self._get_exact_match_key(eff_namespace, prompt)
            with self._exact_match_lock:
                if key in self._exact_match_index:
                    del self._exact_match_index[key]
        else:
            # invalidate entire namespace
            deleted = self.store.delete(eff_namespace)
            
            # remove from exact_match_index
            with self._exact_match_lock:
                keys_to_del = [k for k in self._exact_match_index.keys() if self._exact_match_index.get("namespace_hint") == eff_namespace]
                # this is imprecise as we don't store namespace clearly in exact_match_index key, 
                # but we can just clear it fully or ignore. Realistically, we'd store namespace inside the dict.
                # Let's just clear the whole exact index for safety if they invalidate a namespace.
                self._exact_match_index.clear()
                
            logger.info(f"Invalidated {deleted} entries in namespace {eff_namespace}")

    def serve_dashboard(self, port: int = 5555, background: bool = True):
        if not self._dashboard_enabled:
            raise ConfigurationError("Dashboard is not enabled. Pass dashboard=True to SemanticCache().")
            
        def run_dashboard():
            import fastcache.dashboard.app as app_module
            # Pass the current cache instance directly to the dashboard
            app_module._global_cache = self
            try:
                app_module.run_server(port)
            except OSError as e:
                # If the port is already in use, we assume the server is already running
                logger.debug(f"Dashboard server already running on port {port}: {e}")
            
        if background:
            t = threading.Thread(target=run_dashboard, daemon=True)
            t.start()
            logger.info(f"Dashboard started in background thread on port {port}")
        else:
            logger.info(f"Dashboard starting on port {port} (blocking)")
            run_dashboard()
