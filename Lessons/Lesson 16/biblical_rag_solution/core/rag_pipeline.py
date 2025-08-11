"""
Enhanced RAG pipeline supporting multiple retriever types.
"""
import logging
from typing import Any, Dict, List, Union

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..interfaces import BaseRetrieverInterface
from ..retrievers import TfidfRetriever
from ..retrievers.factory import RetrieverFactory

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RagPipeline:
    """
    Enhanced RAG pipeline supporting multiple retriever types (TF-IDF, semantic, hybrid).
    Maintains backward compatibility while adding new capabilities.
    """
    
    def __init__(self, retriever: Union[BaseRetrieverInterface, TfidfRetriever] = None, 
                 llm: ChatGoogleGenerativeAI = None, retriever_factory: RetrieverFactory = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance (backwards compatibility)
            llm: Language model for generation
            retriever_factory: Factory for creating different retrievers
        """
        self.llm = llm
        self.retriever_factory = retriever_factory
        self.current_retriever = retriever
        self.current_retriever_type = "tfidf" if isinstance(retriever, TfidfRetriever) else "unknown"
        
        # Create a prompt template for RAG with citations and language-aware responses
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about Biblical texts. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

When providing your answer, please include specific citations to the sources you used.

Context:
{context}

Question: {question}

IMPORTANT: Please respond in the same language as the question. If the question is in Hebrew, respond in Hebrew. If in English, respond in English.

Answer:""")
        
        logging.info(f"RAG Pipeline initialized with retriever type: {self.current_retriever_type}")

    def set_retriever(self, retriever_type: str, **kwargs) -> None:
        """
        Set the active retriever type.
        
        Args:
            retriever_type: Type of retriever ("tfidf", "semantic", "hybrid")
            **kwargs: Additional configuration for the retriever
        """
        if self.retriever_factory is None:
            raise RuntimeError("RetrieverFactory not provided. Cannot switch retriever types.")
        
        logging.info(f"Switching to {retriever_type} retriever...")
        self.current_retriever = self.retriever_factory.create_retriever(retriever_type, **kwargs)
        self.current_retriever_type = retriever_type
        logging.info(f"Successfully switched to {retriever_type} retriever")

    def answer_question(self, question: str, retriever_type: str = None, 
                       top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The user's question
            retriever_type: Optional retriever type to use for this query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for retrieval

        Returns:
            Dictionary containing the answer and source documents
        """
        logging.info(f"Answering question: '{question}' using RAG pipeline")
        
        # Switch retriever if requested
        if retriever_type and retriever_type != self.current_retriever_type:
            self.set_retriever(retriever_type, **kwargs)
        
        if self.current_retriever is None:
            raise RuntimeError("No retriever configured. Please set a retriever first.")
        
        # Get relevant documents
        if hasattr(self.current_retriever, 'get_similarity_scores'):
            # Use enhanced interface with scores
            docs_with_scores = self.current_retriever.get_similarity_scores(question, top_k)
            relevant_docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
        else:
            # Fallback to basic retrieval
            relevant_docs = self.current_retriever.retrieve(question, top_k)
            scores = [0.0] * len(relevant_docs)  # No scores available
        
        # Format context
        context = self._format_docs(relevant_docs)
        
        # Generate answer using language-aware prompt
        prompt_text = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(prompt_text).content
        
        logging.info(f"Successfully generated answer using {self.current_retriever_type} retrieval")
        
        return {
            "answer": answer,
            "sources": relevant_docs,
            "scores": scores,
            "retriever_type": self.current_retriever_type,
            "num_sources": len(relevant_docs)
        }

    def compare_retrievers(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Compare results from different retriever types for the same question.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve from each method
            
        Returns:
            Dictionary with results from each retriever type
        """
        if self.retriever_factory is None:
            raise RuntimeError("RetrieverFactory not provided. Cannot compare retrievers.")
        
        results = {}
        
        for retriever_type in ["tfidf", "semantic", "hybrid"]:
            try:
                logging.info(f"Getting results from {retriever_type} retriever...")
                result = self.answer_question(question, retriever_type, top_k)
                results[retriever_type] = result
            except Exception as e:
                logging.error(f"Error with {retriever_type} retriever: {e}")
                results[retriever_type] = {
                    "error": str(e),
                    "answer": f"Error using {retriever_type} retrieval",
                    "sources": [],
                    "scores": []
                }
        
        return results

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the current retriever."""
        info = {
            "type": self.current_retriever_type,
            "available_types": ["tfidf", "semantic", "hybrid"] if self.retriever_factory else [self.current_retriever_type]
        }
        
        if hasattr(self.current_retriever, 'get_corpus_size'):
            info["corpus_size"] = self.current_retriever.get_corpus_size()
        
        if hasattr(self.current_retriever, 'get_embedding_dimension'):
            info["embedding_dimension"] = self.current_retriever.get_embedding_dimension()
        
        return info

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context."""
        formatted_docs = []
        for doc in docs:
            source_info = self._format_source_info(doc.metadata)
            formatted_docs.append(f"Source: {source_info}\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)

    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """Format source information from metadata."""
        book = metadata.get('book', 'Unknown')
        chapter = metadata.get('chapter', '')
        verse = metadata.get('verse', '')
        
        if chapter and verse:
            return f"{book.title()} {chapter}:{verse}"
        elif chapter:
            return f"{book.title()} Chapter {chapter}"
        else:
            return book.title()

    # Backwards compatibility methods
    def retrieve_documents(self, question: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using the current retriever (backwards compatibility)."""
        if hasattr(self.current_retriever, 'retrieve'):
            return self.current_retriever.retrieve(question, top_k)
        else:
            return self.current_retriever._get_relevant_documents(question, top_k=top_k)