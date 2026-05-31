class FastCacheError(Exception):
    """Base exception for all FastCache errors."""
    def __init__(self, message: str, suggestion: str = "", cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.__cause__ = cause
        
    def __str__(self):
        msg = self.message
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        if self.__cause__:
            msg += f"\nCaused by: {self.__cause__}"
        return msg

class ConfigurationError(FastCacheError):
    """Raised for bad or missing config, such as missing API keys."""
    pass

class EmbedderError(FastCacheError):
    """Raised when embedding fails."""
    pass

class EmbedderAuthError(EmbedderError):
    """Raised when an API key is invalid."""
    pass

class StoreError(FastCacheError):
    """Raised when a vector store operation fails."""
    pass

class StoreConnectionError(StoreError):
    """Raised when the vector store cannot be connected to."""
    pass

class GuardrailError(FastCacheError):
    """Raised when a pre-query validation rejects the prompt."""
    pass

class FallbackError(FastCacheError):
    """Raised when the fallback LLM function raises an exception."""
    pass

class CacheWarmingError(FastCacheError):
    """Raised when warm() or warm_from_csv() fails."""
    pass
