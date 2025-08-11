"""
Factory for creating and managing different types of retrievers.
"""
import logging
import os
from typing import Any, Dict, List

from ..config import Config
from ..interfaces import BaseRetrieverInterface
from ..storage.cache_manager import EmbeddingCacheManager, TFIDFCacheManager
from .hybrid_retriever import HybridRetriever
from .semantic_retriever import SemanticRetriever, build_semantic_index
from .tfidf_retriever import TfidfRetriever, build_tfidf_index

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RetrieverFactory:
    """
    Factory class for creating and managing retrievers with caching support.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the retriever factory.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.data_dir = self.config.DATA_DIR
        
        # Get absolute path for data directory
        if not os.path.isabs(self.data_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(project_root, self.data_dir)
        
        # Cache managers - cache directory is relative to the biblical_rag package
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # biblical_rag directory
        cache_dir = os.path.join(package_dir, self.config.cache.cache_dir)
        self.tfidf_cache = TFIDFCacheManager(cache_dir)
        self.embedding_cache = EmbeddingCacheManager(cache_dir)
        
        # Cached components
        self._tfidf_components = None
        self._semantic_components = None
        
        logging.info(f"RetrieverFactory initialized with data_dir: {self.data_dir}")
    
    def create_retriever(self, retriever_type: str, force_rebuild: bool = False, 
                        **kwargs) -> BaseRetrieverInterface:
        """
        Create a retriever of the specified type.
        
        Args:
            retriever_type: Type of retriever ("tfidf", "semantic", "hybrid")
            force_rebuild: Whether to force rebuilding of indices
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            Configured retriever instance
        """
        if retriever_type == "tfidf":
            return self._create_tfidf_retriever(force_rebuild)
        elif retriever_type == "semantic":
            return self._create_semantic_retriever(force_rebuild, **kwargs)
        elif retriever_type == "hybrid":
            return self._create_hybrid_retriever(force_rebuild, **kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def _create_tfidf_retriever(self, force_rebuild: bool = False) -> TfidfRetriever:
        """Create or load TF-IDF retriever."""
        if self._tfidf_components is None or force_rebuild:
            if not force_rebuild and self.tfidf_cache.is_valid(self.data_dir):
                logging.info("Loading TF-IDF components from cache...")
                vectorizer, matrix, corpus, metadata = self.tfidf_cache.load()
            else:
                logging.info("Building TF-IDF index...")
                vectorizer, matrix, corpus, metadata = build_tfidf_index(self.data_dir, self.config.chunk)
                
                if vectorizer is not None:
                    logging.info("Saving TF-IDF components to cache...")
                    self.tfidf_cache.save(vectorizer, matrix, corpus, metadata, self.data_dir)
                else:
                    raise RuntimeError("Failed to build TF-IDF index")
            
            self._tfidf_components = (vectorizer, matrix, corpus, metadata)
        
        vectorizer, matrix, corpus, metadata = self._tfidf_components
        return TfidfRetriever(vectorizer, matrix, corpus, metadata)
    
    def _create_semantic_retriever(self, force_rebuild: bool = False, **kwargs) -> SemanticRetriever:
        """Create or load semantic retriever."""
        # Get configuration with overrides from kwargs
        model_name = kwargs.get('model_name', self.config.embedding.model_name)
        batch_size = kwargs.get('batch_size', self.config.embedding.batch_size)
        device = kwargs.get('device', self.config.embedding.device)
        
        if self._semantic_components is None or force_rebuild:
            if not force_rebuild and self.embedding_cache.is_valid(self.data_dir):
                logging.info("Loading semantic components from cache...")
                try:
                    embeddings, vector_store_path, metadata = self.embedding_cache.load()
                    
                    # We need to reconstruct the corpus from metadata or load separately
                    # For now, we'll rebuild if we can't load the corpus properly
                    if embeddings is None or metadata is None:
                        raise RuntimeError("Cached semantic components incomplete")
                    
                    # Load the sentence transformer model
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(model_name, device=device)
                    
                    # Reconstruct corpus from cached data
                    corpus = self._reconstruct_corpus_from_metadata(metadata)
                    
                    self._semantic_components = (model, embeddings, corpus, metadata)
                    
                except Exception as e:
                    logging.warning(f"Failed to load semantic cache: {e}. Rebuilding...")
                    force_rebuild = True
            
            if force_rebuild or self._semantic_components is None:
                logging.info("Building semantic index...")
                model, embeddings, corpus, metadata = build_semantic_index(
                    self.data_dir, model_name, batch_size, device, self.config.chunk
                )
                
                if model is not None:
                    logging.info("Saving semantic components to cache...")
                    self.embedding_cache.save(
                        embeddings=embeddings,
                        metadata=metadata,
                        data_dir=self.data_dir
                    )
                else:
                    raise RuntimeError("Failed to build semantic index")
                
                self._semantic_components = (model, embeddings, corpus, metadata)
        
        model, embeddings, corpus, metadata = self._semantic_components
        return SemanticRetriever(model, embeddings, corpus, metadata)
    
    def _create_hybrid_retriever(self, force_rebuild: bool = False, **kwargs) -> HybridRetriever:
        """Create hybrid retriever."""
        # Get configuration with overrides from kwargs
        tfidf_weight = kwargs.get('tfidf_weight', self.config.retrieval.tfidf_weight)
        semantic_weight = kwargs.get('semantic_weight', self.config.retrieval.semantic_weight)
        fusion_method = kwargs.get('fusion_method', 'weighted')
        
        # Create individual retrievers
        tfidf_retriever = self._create_tfidf_retriever(force_rebuild)
        semantic_retriever = self._create_semantic_retriever(force_rebuild, **kwargs)
        
        return HybridRetriever(
            tfidf_retriever=tfidf_retriever,
            semantic_retriever=semantic_retriever,
            tfidf_weight=tfidf_weight,
            semantic_weight=semantic_weight,
            fusion_method=fusion_method
        )
    
    def _reconstruct_corpus_from_metadata(self, metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Reconstruct corpus from metadata by re-reading the data files.
        This is a fallback when corpus is not cached separately.
        """
        from ..core.data_loader import load_all_data_generator
        
        corpus = []
        data_generator = load_all_data_generator(self.data_dir)
        
        # Create a mapping of metadata to find corresponding texts
        metadata_keys = []
        for meta in metadata:
            key = (meta.get('category'), meta.get('source_file'), 
                  meta.get('book'), meta.get('chapter'), meta.get('verse'))
            metadata_keys.append(key)
        
        current_index = 0
        for category, filename, record in data_generator:
            # Extract text
            text = None
            # Fix prioritization: For Tanach, prioritize text_he over text
            if category.lower() == 'tanach' and record.get('text_he') and record.get('text_he').strip():
                text = record.get('text_he')  # Hebrew text for Tanach
            elif record.get('text') and record.get('text').strip():
                text = record.get('text')  # Default text field for Bavli/Mishnah
            elif record.get('text_he') and record.get('text_he').strip():
                text = record.get('text_he')  # Hebrew fallback
            elif record.get('text_en') and record.get('text_en').strip():
                text = record.get('text_en')  # English fallback
            
            if text and current_index < len(metadata_keys):
                # Check if this record matches the expected metadata
                record_key = (category, filename, record.get('book'), 
                             record.get('chapter'), record.get('verse'))
                
                if record_key == metadata_keys[current_index]:
                    corpus.append(text)
                    current_index += 1
        
        if len(corpus) != len(metadata):
            logging.warning(f"Corpus reconstruction mismatch: {len(corpus)} texts vs {len(metadata)} metadata entries")
        
        return corpus
    
    def get_available_retrievers(self) -> List[str]:
        """Get list of available retriever types."""
        return ["tfidf", "semantic", "hybrid"]
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cached components.
        
        Args:
            cache_type: Type of cache to clear ("tfidf", "semantic", "all")
        """
        if cache_type in ["tfidf", "all"]:
            self.tfidf_cache.clear()
            self._tfidf_components = None
            logging.info("TF-IDF cache cleared")
        
        if cache_type in ["semantic", "all"]:
            self.embedding_cache.clear()
            self._semantic_components = None
            logging.info("Semantic cache cleared")
    
    def get_cache_status(self) -> Dict[str, bool]:
        """Get status of different caches."""
        return {
            'tfidf_cache_valid': self.tfidf_cache.is_valid(self.data_dir),
            'embedding_cache_valid': self.embedding_cache.is_valid(self.data_dir)
        }
    
    def preload_components(self, retriever_types: List[str] = None) -> None:
        """
        Preload components for specified retriever types.
        
        Args:
            retriever_types: List of retriever types to preload (default: all)
        """
        if retriever_types is None:
            retriever_types = ["tfidf", "semantic"]
        
        for retriever_type in retriever_types:
            if retriever_type == "tfidf":
                self._create_tfidf_retriever()
            elif retriever_type == "semantic":
                self._create_semantic_retriever()
            
        logging.info(f"Preloaded components for: {retriever_types}")
