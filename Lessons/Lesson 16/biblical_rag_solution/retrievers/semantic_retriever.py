"""
Semantic retriever using sentence transformers for contextual search in Biblical texts.
"""
import logging
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..interfaces import BaseRetrieverInterface

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_semantic_index(data_dir: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                        batch_size: int = 32, device: str = "cpu", chunk_config=None) -> Tuple[SentenceTransformer, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Builds a semantic index using sentence transformers with configurable chunking.

    Args:
        data_dir: Path to the data directory
        model_name: Name of the sentence transformer model
        batch_size: Batch size for embedding computation
        device: Device to use ("cpu" or "cuda")
        chunk_config: Configuration for text chunking

    Returns:
        Tuple containing:
        - The SentenceTransformer model
        - The embeddings matrix (numpy array)
        - List of document texts (corpus)
        - List of metadata dictionaries
    """
    logging.info(f"Starting semantic index build with model: {model_name}")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name, device=device)
    logging.info(f"Loaded sentence transformer model on device: {device}")
    
    corpus = []
    metadata = []
    
    # Use chunked corpus if chunk_config is provided
    if chunk_config:
        from ..core.chunker import create_chunked_corpus
        print("Reading documents and preparing chunked corpus...")
        
        for chunk in create_chunked_corpus(data_dir, chunk_config):
            # Extract text from chunk
            text = chunk.get('text', '').strip()
            
            if text:
                corpus.append(text)
                metadata.append({
                    'category': chunk.get('category'),
                    'source_file': chunk.get('source_file'),
                    'book': chunk.get('book'),
                    'chapter': chunk.get('chapter'),
                    'verse': chunk.get('verse'),
                    'verses': chunk.get('verses', []),
                    'chunk_type': chunk.get('chunk_type'),
                    'chunk_size': chunk.get('chunk_size'),
                    'text_en': chunk.get('text_en', ''),
                    'text_pisuk': chunk.get('text_pisuk', '')
                })
            else:
                logging.warning(f"Skipping chunk with no text in {chunk.get('source_file')}")
    else:
        # Original verse-by-verse processing
        from ..core.data_loader import load_all_data_generator
        data_generator = load_all_data_generator(data_dir)
        print("Reading documents and preparing corpus...")
        
        for category, filename, record in data_generator:
            # Extract text from record
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
            
            if text:
                corpus.append(text)
                metadata.append({
                    'category': category,
                    'source_file': filename,
                    'book': record.get('book'),
                    'chapter': record.get('chapter'),
                    'verse': record.get('verse')
                })
            else:
                available_fields = list(record.keys())
                logging.warning(f"Skipping record with no text fields in {filename}. Available fields: {available_fields}")
    
    if not corpus:
        logging.error("No documents found to build the semantic index. Aborting.")
        return None, None, None, None
    
    logging.info(f"Successfully prepared a corpus of {len(corpus)} documents.")
    
    # Generate embeddings in batches
    logging.info("Computing semantic embeddings...")
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    logging.info(f"Semantic embeddings computed. Shape: {embeddings.shape}")
    
    return model, embeddings, corpus, metadata


class SemanticRetriever(BaseRetrieverInterface):
    """
    A retriever that uses semantic similarity via sentence transformers.
    """
    
    def __init__(self, model: SentenceTransformer, embeddings: np.ndarray, 
                 corpus: List[str], metadata: List[Dict[str, Any]]):
        self.model = model
        self.embeddings = embeddings
        self.corpus = corpus
        self.metadata = metadata
        
        # Initialize FAISS index for efficient similarity search
        if len(embeddings) > 0:
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            self.faiss_index.add(embeddings_normalized.astype('float32'))
            # Store normalized embeddings for consistency
            self.embeddings_normalized = embeddings_normalized
            logging.info(f"FAISS index initialized with {len(embeddings)} vectors, dimension {dimension}")
        else:
            self.faiss_index = None
            self.embeddings_normalized = np.array([])
        
        logging.info("SemanticRetriever initialized.")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents using semantic similarity.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters (unused for semantic search)
            
        Returns:
            List of LangChain Document objects
        """
        docs_with_scores = self.get_similarity_scores(query, top_k)
        return [doc for doc, _ in docs_with_scores]
    
    def get_similarity_scores(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Get documents with their semantic similarity scores using FAISS.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if not query.strip():
            return []
        
        logging.info(f"Computing semantic similarity scores for query: '{query}' with top_k={top_k}")
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        if self.faiss_index is not None:
            # Use FAISS for efficient similarity search
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search using FAISS
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k, len(self.corpus)))
            
            # Build results with documents and scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.corpus) and idx < len(self.metadata):  # Valid index with bounds check
                    doc = Document(
                        page_content=self.corpus[idx],
                        metadata=self.metadata[idx]
                    )
                    results.append((doc, float(score)))
                elif idx >= 0:
                    logging.warning(f"Index {idx} out of bounds: corpus={len(self.corpus)}, metadata={len(self.metadata)}")
        else:
            # Fallback to sklearn if no FAISS index
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # Get top-k most similar documents
            if len(similarities) <= top_k:
                top_indices = range(len(similarities))
            else:
                top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Build results with documents and scores
            results = []
            for idx in top_indices:
                if idx < len(self.corpus) and idx < len(self.metadata):  # Bounds check
                    doc = Document(
                        page_content=self.corpus[idx],
                        metadata=self.metadata[idx]
                    )
                    score = float(similarities[idx])
                    results.append((doc, score))
                else:
                    logging.warning(f"Index {idx} out of bounds: corpus={len(self.corpus)}, metadata={len(self.metadata)}")
        
        logging.info(f"Retrieved {len(results)} documents with semantic scores.")
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
    
    def get_corpus_size(self) -> int:
        """Get the size of the corpus."""
        return len(self.corpus)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Add new documents to the index (useful for incremental updates).
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
        """
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")
        
        logging.info(f"Adding {len(texts)} documents to semantic index...")
        
        # Generate embeddings for new texts
        new_embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Append to existing data
        self.corpus.extend(texts)
        self.metadata.extend(metadatas)
        
        # Concatenate embeddings
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update FAISS index
        if self.faiss_index is not None:
            # Normalize new embeddings and add to FAISS index
            new_embeddings_normalized = new_embeddings.copy()
            faiss.normalize_L2(new_embeddings_normalized)
            self.faiss_index.add(new_embeddings_normalized.astype('float32'))
            
            # Update normalized embeddings
            if len(self.embeddings_normalized) == 0:
                self.embeddings_normalized = new_embeddings_normalized
            else:
                self.embeddings_normalized = np.vstack([self.embeddings_normalized, new_embeddings_normalized])
        
        logging.info(f"Added {len(texts)} documents. Total corpus size: {len(self.corpus)}")
    
    def save_embeddings(self, filepath: str) -> None:
        """Save embeddings to a file."""
        np.save(filepath, self.embeddings)
        logging.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> None:
        """Load embeddings from a file."""
        self.embeddings = np.load(filepath)
        logging.info(f"Embeddings loaded from {filepath}")
    
    @classmethod
    def from_cache(cls, model_name: str, embeddings_path: str, corpus: List[str], 
                   metadata: List[Dict[str, Any]], device: str = "cpu") -> 'SemanticRetriever':
        """
        Create a SemanticRetriever from cached embeddings.
        
        Args:
            model_name: Name of the sentence transformer model
            embeddings_path: Path to the cached embeddings file
            corpus: List of document texts
            metadata: List of metadata dictionaries
            device: Device to load the model on
            
        Returns:
            SemanticRetriever instance
        """
        model = SentenceTransformer(model_name, device=device)
        embeddings = np.load(embeddings_path)
        
        return cls(model, embeddings, corpus, metadata)
