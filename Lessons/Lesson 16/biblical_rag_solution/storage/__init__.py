"""
Storage module for the Biblical RAG system.
"""
from .cache_manager import EmbeddingCacheManager  # Backwards compatibility
from .cache_manager import (TFIDFCacheManager, get_cache_manager,
                            is_cache_valid, load_from_cache, save_to_cache)

__all__ = [
    'TFIDFCacheManager',
    'EmbeddingCacheManager', 
    'get_cache_manager',
    'is_cache_valid',
    'save_to_cache',
    'load_from_cache'
]
