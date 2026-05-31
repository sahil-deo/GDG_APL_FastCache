import os
from fastcache import SemanticCache

# Disable network embedder for basic offline testing if no key provided.
# We'll use a dummy embedder for this example.
from fastcache.embedders.base import BaseEmbedder
import numpy as np

class DummyEmbedder(BaseEmbedder):
    @property
    def dimension(self):
        return 4
        
    def embed(self, text):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

cache = SemanticCache(embedder=DummyEmbedder())

def my_llm(prompt: str) -> str:
    return f"Answer to: {prompt}"

print("Calling for the first time...")
response1 = cache.query("What is Python?", my_llm)
print("Response:", response1)

print("\nCalling exact match...")
response2 = cache.query("What is Python?", my_llm)
print("Response:", response2)

print("\nStats:")
print(cache.stats)
