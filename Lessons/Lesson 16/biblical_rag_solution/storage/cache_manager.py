"""
Enhanced cache management system supporting TF-IDF and semantic embeddings.
"""
import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from ..interfaces import CacheManagerInterface

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseCacheManager(CacheManagerInterface):
    """Base cache manager with common functionality."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.ensure_cache_dir_exists()
    
    def ensure_cache_dir_exists(self):
        """Creates the cache directory if it doesn't already exist."""
        if not os.path.exists(self.cache_dir):
            logging.info(f"Cache directory not found. Creating it at: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_data_hash(self, data_dir: str) -> str:
        """Generate a hash of the data directory to detect changes."""
        hash_md5 = hashlib.md5()
        
        for root, dirs, files in os.walk(data_dir):
            # Sort to ensure consistent ordering
            dirs.sort()
            files.sort()
            
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        # Hash file path and modification time
                        hash_md5.update(file_path.encode('utf-8'))
                        hash_md5.update(str(os.path.getmtime(file_path)).encode('utf-8'))
                    except OSError:
                        continue
        
        return hash_md5.hexdigest()
    
    def _is_cache_expired(self, cache_file: str, timeout_hours: int = 168) -> bool:
        """Check if cache file is expired based on modification time."""
        if not os.path.exists(cache_file):
            return True
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - file_time > timedelta(hours=timeout_hours)


class TFIDFCacheManager(BaseCacheManager):
    """Cache manager specifically for TF-IDF components."""
    
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.vectorizer_path = os.path.join(cache_dir, 'tfidf_vectorizer.pkl')
        self.matrix_path = os.path.join(cache_dir, 'tfidf_matrix.pkl')
        self.corpus_path = os.path.join(cache_dir, 'corpus.pkl')
        self.metadata_path = os.path.join(cache_dir, 'metadata.pkl')
        self.hash_path = os.path.join(cache_dir, 'data_hash.txt')
    
    def is_valid(self, data_dir: str) -> bool:
        """Check if TF-IDF cache is valid for the given data directory."""
        required_files = [
            self.vectorizer_path, self.matrix_path, 
            self.corpus_path, self.metadata_path, self.hash_path
        ]
        
        # Check if all cache files exist
        if not all(os.path.exists(f) for f in required_files):
            logging.info("TF-IDF cache incomplete - missing files")
            return False
        
        # Check if cache is expired
        if any(self._is_cache_expired(f) for f in required_files):
            logging.info("TF-IDF cache expired")
            return False
        
        # Check if data has changed
        try:
            with open(self.hash_path, 'r') as f:
                cached_hash = f.read().strip()
            current_hash = self._get_data_hash(data_dir)
            
            if cached_hash != current_hash:
                logging.info("TF-IDF cache invalid - data changed")
                return False
        except (IOError, OSError):
            logging.info("TF-IDF cache invalid - hash file unreadable")
            return False
        
        return True
    
    def save(self, vectorizer: Any, matrix: Any, corpus: List[str], 
             metadata: List[Dict[str, Any]], data_dir: str) -> None:
        """Save TF-IDF components to cache."""
        try:
            logging.info("Saving TF-IDF vectorizer to cache...")
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

            logging.info("Saving TF-IDF matrix to cache...")
            with open(self.matrix_path, 'wb') as f:
                pickle.dump(matrix, f)

            logging.info("Saving corpus to cache...")
            with open(self.corpus_path, 'wb') as f:
                pickle.dump(corpus, f)

            logging.info("Saving metadata to cache...")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save data hash
            with open(self.hash_path, 'w') as f:
                f.write(self._get_data_hash(data_dir))

            logging.info("✅ TF-IDF cache saved successfully.")
            
        except Exception as e:
            logging.error(f"❌ Error saving TF-IDF cache: {e}")
            raise
    
    def load(self) -> Tuple[Any, Any, List[str], List[Dict[str, Any]]]:
        """Load TF-IDF components from cache."""
        try:
            logging.info("Loading TF-IDF components from cache...")
            
            with open(self.vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            with open(self.matrix_path, 'rb') as f:
                matrix = pickle.load(f)
            
            with open(self.corpus_path, 'rb') as f:
                corpus = pickle.load(f)
            
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            logging.info("✅ TF-IDF cache loaded successfully.")
            return vectorizer, matrix, corpus, metadata
            
        except Exception as e:
            logging.error(f"❌ Error loading TF-IDF cache: {e}")
            raise
    
    def clear(self) -> None:
        """Clear TF-IDF cache files."""
        cache_files = [
            self.vectorizer_path, self.matrix_path,
            self.corpus_path, self.metadata_path, self.hash_path
        ]
        
        for file_path in cache_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed cache file: {file_path}")
            except OSError as e:
                logging.warning(f"Could not remove cache file {file_path}: {e}")


class EmbeddingCacheManager(BaseCacheManager):
    """Cache manager for semantic embeddings and vector stores."""
    
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.embeddings_path = os.path.join(cache_dir, 'embeddings.pkl')
        self.vector_store_path = os.path.join(cache_dir, 'vector_store')
        self.embedding_metadata_path = os.path.join(cache_dir, 'embedding_metadata.pkl')
        self.embedding_hash_path = os.path.join(cache_dir, 'embedding_hash.txt')
    
    def is_valid(self, data_dir: str) -> bool:
        """Check if embedding cache is valid for the given data directory."""
        required_files = [
            self.embeddings_path, self.embedding_metadata_path, self.embedding_hash_path
        ]
        
        # Check if all cache files exist
        if not all(os.path.exists(f) for f in required_files):
            logging.info("Embedding cache incomplete - missing files")
            return False
        
        # Check if cache is expired
        if any(self._is_cache_expired(f) for f in required_files):
            logging.info("Embedding cache expired")
            return False
        
        # Check if data has changed
        try:
            with open(self.embedding_hash_path, 'r') as f:
                cached_hash = f.read().strip()
            current_hash = self._get_data_hash(data_dir)
            
            if cached_hash != current_hash:
                logging.info("Embedding cache invalid - data changed")
                return False
        except (IOError, OSError):
            logging.info("Embedding cache invalid - hash file unreadable")
            return False
        
        return True
    
    def save(self, embeddings: Any = None, vector_store: Any = None, 
             metadata: List[Dict[str, Any]] = None, data_dir: str = None) -> None:
        """Save embedding components to cache."""
        try:
            if embeddings is not None:
                logging.info("Saving embeddings to cache...")
                with open(self.embeddings_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            
            if vector_store is not None:
                logging.info("Saving vector store to cache...")
                vector_store.save(self.vector_store_path)
            
            if metadata is not None:
                logging.info("Saving embedding metadata to cache...")
                with open(self.embedding_metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            
            if data_dir is not None:
                # Save data hash
                with open(self.embedding_hash_path, 'w') as f:
                    f.write(self._get_data_hash(data_dir))

            logging.info("✅ Embedding cache saved successfully.")
            
        except Exception as e:
            logging.error(f"❌ Error saving embedding cache: {e}")
            raise
    
    def load(self) -> Tuple[Any, Any, List[Dict[str, Any]]]:
        """Load embedding components from cache."""
        try:
            logging.info("Loading embedding components from cache...")
            
            embeddings = None
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
            
            metadata = None
            if os.path.exists(self.embedding_metadata_path):
                with open(self.embedding_metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            logging.info("✅ Embedding cache loaded successfully.")
            return embeddings, self.vector_store_path, metadata
            
        except Exception as e:
            logging.error(f"❌ Error loading embedding cache: {e}")
            raise
    
    def clear(self) -> None:
        """Clear embedding cache files."""
        cache_files = [
            self.embeddings_path, self.embedding_metadata_path, self.embedding_hash_path
        ]
        
        for file_path in cache_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed cache file: {file_path}")
            except OSError as e:
                logging.warning(f"Could not remove cache file {file_path}: {e}")
        
        # Remove vector store directory
        import shutil
        if os.path.exists(self.vector_store_path):
            try:
                shutil.rmtree(self.vector_store_path)
                logging.info(f"Removed vector store directory: {self.vector_store_path}")
            except OSError as e:
                logging.warning(f"Could not remove vector store directory: {e}")


# Factory function for backwards compatibility
def get_cache_manager(cache_type: str = "tfidf", cache_dir: str = "cache") -> BaseCacheManager:
    """Factory function to get appropriate cache manager."""
    if cache_type == "tfidf":
        return TFIDFCacheManager(cache_dir)
    elif cache_type == "embedding":
        return EmbeddingCacheManager(cache_dir)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# Backwards compatibility functions
def is_cache_valid(data_dir: str) -> bool:
    """Check if TF-IDF cache is valid - backwards compatibility function."""
    cache_manager = TFIDFCacheManager("cache")
    return cache_manager.is_valid(data_dir)


def save_to_cache(vectorizer: Any, matrix: Any, corpus: List[str], 
                  metadata: List[Dict[str, Any]], data_dir: str = None) -> None:
    """Save to TF-IDF cache - backwards compatibility function."""
    cache_manager = TFIDFCacheManager("cache")
    if data_dir is None:
        # Try to determine data_dir from project structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
    cache_manager.save(vectorizer, matrix, corpus, metadata, data_dir)


def load_from_cache() -> Tuple[Any, Any, List[str], List[Dict[str, Any]]]:
    """Load from TF-IDF cache - backwards compatibility function."""
    cache_manager = TFIDFCacheManager("cache")
    return cache_manager.load()
    return cache_manager.load()
