
"""
Configuration management for the Biblical RAG hybrid retrieval system.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_by: str = "balanced"  # "verse", "chapter", "multi_verse", "balanced"
    verses_per_chunk: int = 100  # Only used when chunk_by = "multi_verse"
    chunk_overlap: int = 10  # Number of overlapping verses between chunks
    include_context: bool = True  # Include surrounding context in metadata


@dataclass
class RetrievalConfig:
    """Configuration for retrieval methods."""
    tfidf_weight: float = 0.7
    semantic_weight: float = 0.3
    default_top_k: int = 5
    max_top_k: int = 20
    
    
@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # "cuda" if available
    

@dataclass
class CacheConfig:
    """Configuration for caching."""
    cache_dir: str = ".cache"
    enable_tfidf_cache: bool = True
    enable_embedding_cache: bool = True
    cache_timeout_hours: int = 168  # 1 week
    

@dataclass
class UIConfig:
    """Configuration for the Gradio interface."""
    default_retrieval_method: str = "hybrid"  # "tfidf", "semantic", "hybrid"
    show_similarity_scores: bool = True
    show_advanced_options: bool = False
    

class Config:
    """Main configuration class for the Biblical RAG system."""
    
    # Environment variables
    DATA_DIR = os.getenv("DATA_DIR", "data")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Component configurations
    chunk = ChunkConfig()
    retrieval = RetrievalConfig()
    embedding = EmbeddingConfig()
    cache = CacheConfig()
    ui = UIConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        if 'chunk' in config_dict:
            config.chunk = ChunkConfig(**config_dict['chunk'])
        if 'retrieval' in config_dict:
            config.retrieval = RetrievalConfig(**config_dict['retrieval'])
        if 'embedding' in config_dict:
            config.embedding = EmbeddingConfig(**config_dict['embedding'])
        if 'cache' in config_dict:
            config.cache = CacheConfig(**config_dict['cache'])
        if 'ui' in config_dict:
            config.ui = UIConfig(**config_dict['ui'])
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'chunk': self.chunk.__dict__,
            'retrieval': self.retrieval.__dict__,
            'embedding': self.embedding.__dict__,
            'cache': self.cache.__dict__,
            'ui': self.ui.__dict__
        }
