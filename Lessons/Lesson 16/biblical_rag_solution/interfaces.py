"""
Base interfaces for the Biblical RAG retrieval system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


class BaseRetrieverInterface(ABC):
    """
    Abstract base class for all retrieval methods in the Biblical RAG system.
    Provides a consistent interface for TF-IDF, semantic, and hybrid retrievers.
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters specific to the retriever
            
        Returns:
            List of LangChain Document objects with relevance scores
        """
        pass
    
    @abstractmethod
    def get_similarity_scores(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Get documents with their similarity scores.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        pass


class CacheManagerInterface(ABC):
    """
    Interface for cache management across different storage types.
    """
    
    @abstractmethod
    def is_valid(self, data_dir: str) -> bool:
        """Check if cache is valid for the given data directory."""
        pass
    
    @abstractmethod
    def save(self, **kwargs) -> None:
        """Save data to cache."""
        pass
    
    @abstractmethod
    def load(self) -> Tuple[Any, ...]:
        """Load data from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the cache."""
        pass


class VectorStoreInterface(ABC):
    """
    Interface for vector storage systems (FAISS, Chroma, etc.).
    """
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add texts with metadata to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass
        pass
