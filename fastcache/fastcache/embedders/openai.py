import os
import numpy as np
from typing import Optional

from fastcache.embedders.base import BaseEmbedder
from fastcache.exceptions import ConfigurationError, EmbedderError

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self._api_key = api_key or os.environ.get("FASTCACHE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ConfigurationError(
                message="OpenAIEmbedder requires an API key but none was found.",
                suggestion="Set the OPENAI_API_KEY environment variable or pass api_key= to OpenAIEmbedder()."
            )
        
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAIEmbedder requires the openai package. Install it with: pip install fastcache[openai]")
        
        self.client = openai.OpenAI(api_key=self._api_key)
        self.model = model
        
        # Dimensions based on model. For 3-small it is 1536 by default.
        self._dimension = 1536 if "small" in model or "ada" in model else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            raise EmbedderError(
                message=f"Failed to embed text with OpenAIEmbedder: {str(e)}",
                suggestion="Check your OpenAI API key and network connection.",
                cause=e
            )
