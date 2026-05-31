from abc import ABC, abstractmethod
import numpy as np
import asyncio

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
        return await asyncio.to_thread(self.embed, text)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension. Used for store validation."""
        ...
