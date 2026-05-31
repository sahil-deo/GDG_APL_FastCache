import os
import uuid
import time
import numpy as np
from typing import Optional

from fastcache.stores.base import BaseVectorStore
from fastcache.models import CacheEntry, LookupResult
from fastcache.exceptions import StoreConnectionError

class RedisStore(BaseVectorStore):
    def __init__(self, url: Optional[str] = None, key_prefix: str = "fastcache", use_vector_sets: bool = True):
        self.url = url or os.environ.get("FASTCACHE_REDIS_URL", "redis://localhost:6379")
        self.key_prefix = key_prefix
        
        try:
            import redis
        except ImportError:
            raise ImportError("RedisStore requires the redis package. Install it with: pip install fastcache[redis]")
            
        try:
            self.client = redis.Redis.from_url(self.url, decode_responses=False)
            self.client.ping()
        except Exception as e:
            raise StoreConnectionError(
                message=f"Cannot connect to Redis at {self.url}.",
                suggestion="Ensure Redis is running or check your FASTCACHE_REDIS_URL.",
                cause=e
            )

    def _entry_key(self, namespace: str, entry_id: str) -> str:
        return f"{self.key_prefix}:{namespace}:entry:{entry_id}"

    def store(
        self,
        vector: np.ndarray,
        query: str,
        response: str,
        namespace: str,
        ttl: int,
    ) -> CacheEntry:
        entry_id = str(uuid.uuid4())
        key = self._entry_key(namespace, entry_id)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        created_at = time.time()
        
        mapping = {
            b"id": entry_id.encode("utf-8"),
            b"query": query.encode("utf-8"),
            b"response": response.encode("utf-8"),
            b"vector": vector.tobytes(),
            b"namespace": namespace.encode("utf-8"),
            b"created_at": str(created_at).encode("utf-8"),
            b"ttl": str(ttl).encode("utf-8"),
            b"hit_count": b"0",
            b"last_accessed": str(created_at).encode("utf-8"),
        }
        
        self.client.hset(key, mapping=mapping)
        if ttl > 0:
            self.client.expire(key, ttl)
            
        return CacheEntry(
            id=entry_id,
            query=query,
            response=response,
            vector=vector,
            namespace=namespace,
            created_at=created_at,
            ttl=ttl,
        )

    def search(
        self,
        vector: np.ndarray,
        namespace: str,
        threshold: float,
    ) -> LookupResult:
        # Fallback manual cosine search
        pattern = f"{self.key_prefix}:{namespace}:entry:*"
        keys = self.client.keys(pattern)
        
        if not keys:
            return LookupResult(hit=False, similarity=0.0, entry=None)
            
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        best_similarity = 0.0
        best_entry_data = None
        
        # In a production Redis system without RediSearch, we fetch all and compute in numpy
        # Pipeline fetching all hashes
        pipe = self.client.pipeline()
        for k in keys:
            pipe.hgetall(k)
        results = pipe.execute()
        
        valid_entries = []
        vectors = []
        
        for data in results:
            if not data:
                continue
            v_bytes = data[b"vector"]
            v = np.frombuffer(v_bytes, dtype=np.float32)
            vectors.append(v)
            valid_entries.append(data)
            
        if not vectors:
            return LookupResult(hit=False, similarity=0.0, entry=None)
            
        matrix = np.stack(vectors)
        similarities = np.dot(matrix, vector)
        
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        
        if best_sim >= threshold:
            best_data = valid_entries[best_idx]
            entry = CacheEntry(
                id=best_data[b"id"].decode("utf-8"),
                query=best_data[b"query"].decode("utf-8"),
                response=best_data[b"response"].decode("utf-8"),
                vector=vectors[best_idx],
                namespace=best_data[b"namespace"].decode("utf-8"),
                created_at=float(best_data[b"created_at"].decode("utf-8")),
                ttl=int(best_data[b"ttl"].decode("utf-8")),
                hit_count=int(best_data.get(b"hit_count", b"0").decode("utf-8")),
                last_accessed=float(best_data.get(b"last_accessed", b"0").decode("utf-8")),
            )
            return LookupResult(hit=True, similarity=best_sim, entry=entry)
            
        return LookupResult(hit=False, similarity=best_sim, entry=None)

    def delete(self, namespace: str, entry_id: Optional[str] = None) -> int:
        if entry_id:
            key = self._entry_key(namespace, entry_id)
            return self.client.delete(key)
            
        pattern = f"{self.key_prefix}:{namespace}:entry:*"
        keys = self.client.keys(pattern)
        if not keys:
            return 0
        return self.client.delete(*keys)

    def size(self, namespace: Optional[str] = None) -> int:
        if namespace:
            pattern = f"{self.key_prefix}:{namespace}:entry:*"
            return len(self.client.keys(pattern))
            
        pattern = f"{self.key_prefix}:*:entry:*"
        return len(self.client.keys(pattern))
