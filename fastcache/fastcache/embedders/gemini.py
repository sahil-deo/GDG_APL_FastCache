import os
import json
import urllib.request
import urllib.error
import numpy as np
from typing import Optional

from fastcache.embedders.base import BaseEmbedder
from fastcache.exceptions import ConfigurationError, EmbedderError, EmbedderAuthError

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        self._api_key = api_key or os.environ.get("FASTCACHE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ConfigurationError(
                message="GeminiEmbedder requires an API key but none was found.",
                suggestion="Set the GEMINI_API_KEY environment variable, or pass api_key= to "
                           "GeminiEmbedder(). Get a free key at https://aistudio.google.com/apikey"
            )
        self.model = model
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self._api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode("utf-8"), 
            headers=headers, 
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                vector = result.get("embedding", {}).get("values")
                if not vector:
                    raise EmbedderError(message="Invalid response from Gemini API: 'embedding.values' missing.", suggestion="Check API documentation.")
                return np.array(vector, dtype=np.float32)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise EmbedderAuthError(
                    message=f"Gemini API authentication failed with status {e.code}.",
                    suggestion="Check that your GEMINI_API_KEY is valid and active.",
                    cause=e
                )
            raise EmbedderError(
                message=f"Gemini API HTTP Error {e.code}: {e.reason}",
                suggestion="Check the Gemini API status or verify your model name.",
                cause=e
            )
        except Exception as e:
            raise EmbedderError(
                message=f"Failed to embed text with GeminiEmbedder: {str(e)}",
                suggestion="Check your network connection and prompt.",
                cause=e
            )
