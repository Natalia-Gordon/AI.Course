"""
Hybrid retriever combining TF-IDF and semantic search with score fusion.
"""
import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from ..interfaces import BaseRetrieverInterface
from .semantic_retriever import SemanticRetriever
from .tfidf_retriever import TfidfRetriever

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRetriever(BaseRetrieverInterface):
    """
    A retriever that combines TF-IDF and semantic search using score fusion.
    Supports multiple fusion strategies including weighted combination and Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, tfidf_retriever: TfidfRetriever, semantic_retriever: SemanticRetriever,
                 tfidf_weight: float = 0.3, semantic_weight: float = 0.7, 
                 fusion_method: str = "weighted"):
        """
        Initialize the hybrid retriever.
        
        Args:
            tfidf_retriever: The TF-IDF retriever instance
            semantic_retriever: The semantic retriever instance
            tfidf_weight: Weight for TF-IDF scores (default 0.3)
            semantic_weight: Weight for semantic scores (default 0.7)
            fusion_method: Method for score fusion ("weighted" or "rrf")
        """
        self.tfidf_retriever = tfidf_retriever
        self.semantic_retriever = semantic_retriever
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight
        self.fusion_method = fusion_method
        
        # Validate weights
        if abs(tfidf_weight + semantic_weight - 1.0) > 1e-6:
            logging.warning(f"TF-IDF and semantic weights don't sum to 1.0: {tfidf_weight} + {semantic_weight} = {tfidf_weight + semantic_weight}")
        
        logging.info(f"HybridRetriever initialized with weights: TF-IDF={tfidf_weight}, Semantic={semantic_weight}, Fusion={fusion_method}")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of LangChain Document objects
        """
        docs_with_scores = self.get_similarity_scores(query, top_k, **kwargs)
        return [doc for doc, _ in docs_with_scores]
    
    def get_similarity_scores(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[Document, float]]:
        """
        Get documents with their hybrid similarity scores.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if not query.strip():
            return []
        
        logging.info(f"Computing hybrid similarity scores for query: '{query}' with top_k={top_k}")
        
        # Get more results from each retriever to ensure good fusion
        retrieval_k = max(top_k * 2, 20)  # Retrieve 2x or at least 20 docs from each method
        
        # Get results from both retrievers
        tfidf_results = self.tfidf_retriever.get_similarity_scores(query, retrieval_k)
        semantic_results = self.semantic_retriever.get_similarity_scores(query, retrieval_k)
        
        # Apply score fusion
        if self.fusion_method == "weighted":
            fused_results = self._weighted_fusion(tfidf_results, semantic_results)
        elif self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(tfidf_results, semantic_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Sort by final score and return top-k
        fused_results.sort(key=lambda x: x[1], reverse=True)
        result = fused_results[:top_k]
        
        logging.info(f"Retrieved {len(result)} documents using hybrid search with {self.fusion_method} fusion.")
        return result
    
    def _weighted_fusion(self, tfidf_results: List[Tuple[Document, float]], 
                        semantic_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Combine scores using weighted fusion.
        
        Args:
            tfidf_results: Results from TF-IDF retriever
            semantic_results: Results from semantic retriever
            
        Returns:
            List of tuples with fused scores
        """
        # Create document content to document mapping for efficient lookup
        doc_scores = {}
        
        # Normalize scores to [0, 1] range for each method
        tfidf_scores = [score for _, score in tfidf_results]
        semantic_scores = [score for _, score in semantic_results]
        
        tfidf_max = max(tfidf_scores) if tfidf_scores else 1.0
        tfidf_min = min(tfidf_scores) if tfidf_scores else 0.0
        semantic_max = max(semantic_scores) if semantic_scores else 1.0
        semantic_min = min(semantic_scores) if semantic_scores else 0.0
        
        # Avoid division by zero
        tfidf_range = max(tfidf_max - tfidf_min, 1e-10)
        semantic_range = max(semantic_max - semantic_min, 1e-10)
        
        # Process TF-IDF results
        for doc, score in tfidf_results:
            normalized_score = (score - tfidf_min) / tfidf_range
            weighted_score = self.tfidf_weight * normalized_score
            doc_key = doc.page_content  # Use content as key for deduplication
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_key]['score'] += weighted_score
        
        # Process semantic results
        for doc, score in semantic_results:
            normalized_score = (score - semantic_min) / semantic_range
            weighted_score = self.semantic_weight * normalized_score
            doc_key = doc.page_content
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_key]['score'] += weighted_score
        
        # Convert back to list of tuples
        return [(item['doc'], item['score']) for item in doc_scores.values()]
    
    def _reciprocal_rank_fusion(self, tfidf_results: List[Tuple[Document, float]], 
                               semantic_results: List[Tuple[Document, float]], k: int = 60) -> List[Tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        RRF formula: score = sum(1 / (k + rank)) for each ranking list
        
        Args:
            tfidf_results: Results from TF-IDF retriever
            semantic_results: Results from semantic retriever
            k: RRF parameter (default 60, commonly used value)
            
        Returns:
            List of tuples with RRF scores
        """
        doc_scores = {}
        
        # Process TF-IDF rankings
        for rank, (doc, _) in enumerate(tfidf_results, 1):
            rrf_score = 1.0 / (k + rank)
            doc_key = doc.page_content
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_key]['score'] += rrf_score
        
        # Process semantic rankings
        for rank, (doc, _) in enumerate(semantic_results, 1):
            rrf_score = 1.0 / (k + rank)
            doc_key = doc.page_content
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {'doc': doc, 'score': 0.0}
            doc_scores[doc_key]['score'] += rrf_score
        
        return [(item['doc'], item['score']) for item in doc_scores.values()]
    
    def update_weights(self, tfidf_weight: float, semantic_weight: float) -> None:
        """
        Update the fusion weights.
        
        Args:
            tfidf_weight: New weight for TF-IDF scores
            semantic_weight: New weight for semantic scores
        """
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight
        
        if abs(tfidf_weight + semantic_weight - 1.0) > 1e-6:
            logging.warning(f"TF-IDF and semantic weights don't sum to 1.0: {tfidf_weight} + {semantic_weight} = {tfidf_weight + semantic_weight}")
        
        logging.info(f"Updated fusion weights: TF-IDF={tfidf_weight}, Semantic={semantic_weight}")
    
    def set_fusion_method(self, method: str) -> None:
        """
        Set the fusion method.
        
        Args:
            method: Fusion method ("weighted" or "rrf")
        """
        if method not in ["weighted", "rrf"]:
            raise ValueError(f"Unknown fusion method: {method}. Use 'weighted' or 'rrf'.")
        
        self.fusion_method = method
        logging.info(f"Set fusion method to: {method}")
    
    def get_individual_results(self, query: str, top_k: int = 5) -> Dict[str, List[Tuple[Document, float]]]:
        """
        Get results from individual retrievers for comparison.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve from each method
            
        Returns:
            Dictionary with results from each retriever
        """
        return {
            'tfidf': self.tfidf_retriever.get_similarity_scores(query, top_k),
            'semantic': self.semantic_retriever.get_similarity_scores(query, top_k),
            'hybrid': self.get_similarity_scores(query, top_k)
        }
