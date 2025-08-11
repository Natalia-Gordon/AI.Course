"""
TF-IDF based retriever for lexical search in Biblical texts.
Refactored to use the new interface system and improved structure.
"""
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.data_loader import load_all_data_generator
from ..interfaces import BaseRetrieverInterface

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_tfidf_index(data_dir: str, chunk_config=None) -> Tuple[Any, Any, List[str], List[Dict[str, Any]]]:
    """
    Builds a TF-IDF index from the data provided by the data loader generator.

    Args:
        data_dir (str): The path to the data directory.
        chunk_config: Configuration for text chunking

    Returns:
        A tuple containing:
        - The trained scikit-learn TfidfVectorizer instance.
        - The computed sparse TF-IDF matrix.
        - A list of document texts (corpus).
        - A list of metadata dictionaries, one for each document.
    """
    logging.info("Starting TF-IDF index build process...")
    
    # The corpus will hold the text of each document for the vectorizer
    corpus = []
    # The metadata list will store the source information for each document
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
        data_generator = load_all_data_generator(data_dir)
        print("Reading documents and preparing corpus...")
        
        for category, filename, record in data_generator:
            # We must have text to process - try different field names
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
                # Show available fields for debugging
                available_fields = list(record.keys())
                logging.warning(f"Skipping record with no text fields ('text', 'text_en', 'text_he') in {filename}. Available fields: {available_fields}")

    if not corpus:
        logging.error("No documents found to build the index. Aborting.")
        return None, None, None, None

    logging.info(f"Successfully prepared a corpus of {len(corpus)} documents.")

    # Initialize and fit the TF-IDF Vectorizer
    # We use sublinear_tf for scaling and analyze at the word level.
    # ngram_range=(1, 2) considers both single words and two-word phrases.
    logging.info("Initializing and fitting the TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english'  # Basic stop word removal
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    logging.info("TF-IDF matrix has been successfully built.")
    logging.info(f"Matrix shape: {tfidf_matrix.shape} (documents, vocabulary size)")

    return vectorizer, tfidf_matrix, corpus, metadata


class TfidfRetriever(BaseRetriever, BaseRetrieverInterface):
    """
    A retriever class that uses a pre-built TF-IDF index to find relevant documents.
    Inherits from both LangChain's BaseRetriever and our custom interface.
    """
    vectorizer: Any
    matrix: Any
    corpus: List[str]
    metadata: List[Dict[str, Any]]
    
    def __init__(self, vectorizer: Any, matrix: Any, corpus: List[str], metadata: List[Dict[str, Any]]):
        super().__init__(
            vectorizer=vectorizer,
            matrix=matrix,
            corpus=corpus,
            metadata=metadata
        )
        logging.info("TfidfRetriever initialized.")

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents using TF-IDF similarity.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters (unused for TF-IDF)
            
        Returns:
            List of LangChain Document objects
        """
        return self._get_relevant_documents(query, top_k=top_k)

    def get_similarity_scores(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Get documents with their TF-IDF similarity scores.
        
        Args:
            query: The search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if not query:
            return []

        logging.info(f"Computing TF-IDF similarity scores for query: '{query}' with top_k={top_k}")
        
        # Transform the query into a TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between the query vector and all document vectors
        cosine_similarities = cosine_similarity(query_vector, self.matrix).flatten()

        # Get the indices of the top_k most similar documents
        if len(cosine_similarities) <= top_k:
            top_indices = range(len(cosine_similarities))
        else:
            top_indices = cosine_similarities.argsort()[-top_k:][::-1]

        # Build the result list with documents and scores
        results = []
        for idx in top_indices:
            # Safety check to prevent index out of range errors
            if idx < len(self.corpus) and idx < len(self.metadata):
                doc = Document(
                    page_content=self.corpus[idx],
                    metadata=self.metadata[idx]
                )
                score = float(cosine_similarities[idx])
                results.append((doc, score))
            else:
                logging.warning(f"Index {idx} out of bounds: corpus={len(self.corpus)}, metadata={len(self.metadata)}")

        logging.info(f"Retrieved {len(results)} documents with TF-IDF scores.")
        return results

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Searches the index for the most relevant documents for a given query.
        (LangChain BaseRetriever interface)

        Args:
            query (str): The user's search query.

        Returns:
            A list of LangChain Document objects.
        """
        if not query:
            return []

        top_k = kwargs.get('top_k', 5)  # Default to 5 if not provided
        logging.info(f"Performing TF-IDF search for query: '{query}' with top_k={top_k}")
        
        # Transform the query into a TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between the query vector and all document vectors
        cosine_similarities = cosine_similarity(query_vector, self.matrix).flatten()

        # Get the indices of the top_k most similar documents
        # We use argpartition for efficiency, which finds the k-th largest values
        # without sorting the entire array.
        if len(cosine_similarities) <= top_k:
            # If we have fewer documents than requested, return all
            top_indices = range(len(cosine_similarities))
        else:
            # Use argpartition for efficiency when we have many documents
            # This gives us the top_k elements but not necessarily sorted
            top_indices = cosine_similarities.argsort()[-top_k:][::-1]  # Sort in descending order

        # Build the result list
        results = []
        for idx in top_indices:
            # Safety check to prevent index out of range errors
            if idx < len(self.corpus) and idx < len(self.metadata):
                # Create a LangChain Document with the text and metadata
                doc = Document(
                    page_content=self.corpus[idx],
                    metadata=self.metadata[idx]
                )
                results.append(doc)
            else:
                logging.warning(f"Index {idx} out of bounds: corpus={len(self.corpus)}, metadata={len(self.metadata)}")

        logging.info(f"Retrieved {len(results)} documents using TF-IDF search.")
        return results
