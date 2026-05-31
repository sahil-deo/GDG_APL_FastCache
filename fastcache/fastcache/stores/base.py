from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import asyncio

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

    async def astore(self, *args, **kwargs) -> CacheEntry:
        return await asyncio.to_thread(self.store, *args, **kwargs)

    async def asearch(self, *args, **kwargs) -> LookupResult:
        return await asyncio.to_thread(self.search, *args, **kwargs)
