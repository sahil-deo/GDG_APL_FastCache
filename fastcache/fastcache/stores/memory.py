import threading
import time
import uuid
import numpy as np
from typing import Optional

from fastcache.stores.base import BaseVectorStore
from fastcache.models import CacheEntry, LookupResult

class InMemoryStore(BaseVectorStore):
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._entries: dict[str, list[CacheEntry]] = {}
        self._lock = threading.RLock()

    def store(
        self,
        vector: np.ndarray,
        query: str,
        response: str,
        namespace: str,
        ttl: int,
    ) -> CacheEntry:
        with self._lock:
            # Enforce max_size globally before adding
            if self.max_size > 0 and self.size() >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                id=str(uuid.uuid4()),
                query=query,
                response=response,
                vector=vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector, # Normalize
                namespace=namespace,
                created_at=time.time(),
                ttl=ttl,
            )
            
            if namespace not in self._entries:
                self._entries[namespace] = []
                
            self._entries[namespace].append(entry)
            return entry

    def search(
        self,
        vector: np.ndarray,
        namespace: str,
        threshold: float,
    ) -> LookupResult:
        with self._lock:
            if namespace not in self._entries or not self._entries[namespace]:
                return LookupResult(hit=False, similarity=0.0, entry=None)
            
            # Lazy cleanup of expired entries
            valid_entries = []
            for entry in self._entries[namespace]:
                if not entry.is_expired:
                    valid_entries.append(entry)
            
            self._entries[namespace] = valid_entries
            
            if not valid_entries:
                return LookupResult(hit=False, similarity=0.0, entry=None)

            # Normalize the query vector for dot product similarity
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            # Batch cosine similarity: dot product of normalized vectors
            matrix = np.stack([e.vector for e in valid_entries])
            similarities = np.dot(matrix, vector)
            
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])
            
            if best_similarity >= threshold:
                return LookupResult(hit=True, similarity=best_similarity, entry=valid_entries[best_idx])
                
            return LookupResult(hit=False, similarity=best_similarity, entry=None)

    def delete(self, namespace: str, entry_id: Optional[str] = None) -> int:
        with self._lock:
            if namespace not in self._entries:
                return 0
                
            if entry_id is None:
                count = len(self._entries[namespace])
                del self._entries[namespace]
                return count
                
            original_len = len(self._entries[namespace])
            self._entries[namespace] = [
                e for e in self._entries[namespace] if e.id != entry_id
            ]
            return original_len - len(self._entries[namespace])

    def size(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is not None:
                return len(self._entries.get(namespace, []))
            return sum(len(ns_entries) for ns_entries in self._entries.values())

    def _evict_lru(self):
        """Evicts the least recently used entry across all namespaces."""
        oldest_entry = None
        oldest_ns = None
        oldest_time = float('inf')
        
        for ns, entries in self._entries.items():
            for entry in entries:
                if entry.last_accessed < oldest_time:
                    oldest_time = entry.last_accessed
                    oldest_entry = entry
                    oldest_ns = ns
                    
        if oldest_entry and oldest_ns:
            self._entries[oldest_ns].remove(oldest_entry)
