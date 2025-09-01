#!/usr/bin/env python3
"""
RAG Service - Main Orchestrator for Hybrid RAG System
Provides clean, maintainable interface for all RAG operations
"""

import time
from typing import List, Dict, Any, Optional
from core.base_agent import BaseAgent, AgentRegistry
from core.config_manager import ConfigManager
from utils.logger import get_logger, log_system_start, log_system_stop, log_performance


class RAGService:
    """Main service for orchestrating RAG operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger('rag_service')
        
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Initialize agent registry
        self.agent_registry = AgentRegistry()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"🚀 RAG Service initialized for project: {self.config.project_name}")
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Import components here to avoid circular imports
            from index.pinecone_index import PineconeIndex
            from index.tfidf_index import TfidfIndexer as TFIDFIndex
            from retrieve.hybrid import HybridRetriever
            from retrieve.rerank import Reranker
            
            # Initialize indices
            self.pinecone_index = PineconeIndex(
                api_key=self.config.pinecone.api_key,
                region=self.config.pinecone.region,
                cloud=self.config.pinecone.cloud,
                index_name=self.config.pinecone.index_name,
                dimension=self.config.embedding.dimension,
                metric=self.config.pinecone.metric
            )
            
            self.tfidf_index = TFIDFIndex()
            
            # Initialize retrieval components
            self.hybrid_retriever = HybridRetriever(
                dense_index=self.pinecone_index,
                sparse_index=self.tfidf_index,
                reranker=Reranker()
            )
            
            self.logger.info("✅ All RAG components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG components: {e}")
            raise
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the service"""
        self.agent_registry.register_agent(agent)
    
    def process_query(self, query: str, document: str, use_langchain: bool = False) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline"""
        start_time = time.time()
        
        # Log query processing start
        log_system_start('rag_service', 
                        query=query,
                        document=document,
                        langchain_enabled=use_langchain)
        
        try:
            # Step 1: Load and process document
            chunks = self._load_document(document)
            
            # Step 2: Build indices
            self._build_indices(chunks)
            
            # Step 3: Route query to appropriate agent
            intent = self._route_query(query)
            
            # Step 4: Execute agent
            result = self._execute_agent(intent, query, chunks)
            
            # Log completion
            duration = time.time() - start_time
            log_performance('rag_service', 'query_processing', duration,
                          chunks_processed=len(chunks),
                          intent=intent,
                          langchain_used=use_langchain)
            
            return {
                'query': query,
                'intent': intent,
                'result': result,
                'chunks_processed': len(chunks),
                'processing_time': duration,
                'agent_stats': self.agent_registry.get_all_stats()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Query processing failed: {e}")
            log_system_stop('rag_service', reason='error', error=str(e))
            raise
    
    def _load_document(self, document: str) -> List[Dict[str, Any]]:
        """Load and process document"""
        try:
            from ingest.data_loader import DataLoader
            
            data_loader = DataLoader(self.config)
            chunks = data_loader.get_processed_chunks(document)
            
            if not chunks:
                self.logger.info(f"Processing document: {document}")
                chunks = data_loader.load_document(document)
                data_loader.save_chunks(chunks, document)
                self.logger.info(f"✅ Document processed: {len(chunks)} chunks created")
            else:
                self.logger.info(f"✅ Loaded {len(chunks)} existing chunks")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load document: {e}")
            raise
    
    def _build_indices(self, chunks: List[Dict[str, Any]]):
        """Build dense and sparse indices"""
        try:
            # Build sparse index
            self.tfidf_index.fit(chunks)
            self.logger.info("✅ Sparse index built successfully")
            
            # Build dense index (Pinecone)
            self.pinecone_index.build_index(chunks)
            self.logger.info("✅ Dense index built successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build indices: {e}")
            raise
    
    def _route_query(self, query: str) -> str:
        """Route query to appropriate agent"""
        try:
            from agents.router import route_intent
            intent = route_intent(query)
            self.logger.info(f"🔀 Query routed to: {intent}")
            return intent
            
        except Exception as e:
            self.logger.warning(f"⚠️ Query routing failed: {e}, defaulting to summary")
            return "summary"
    
    def _execute_agent(self, intent: str, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Execute the appropriate agent for the query"""
        try:
            # Get agent from registry
            agent = self.agent_registry.get_agent(intent)
            if not agent:
                raise ValueError(f"No agent registered for intent: {intent}")
            
            # Execute agent
            result = agent.execute(query, chunks)
            self.logger.info(f"✅ {intent} agent executed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Agent execution failed: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'project_name': self.config.project_name,
            'agents': self.agent_registry.get_all_stats(),
            'indices': {
                'pinecone': self.pinecone_index.get_stats() if hasattr(self.pinecone_index, 'get_stats') else {},
                'tfidf': self.tfidf_index.get_stats() if hasattr(self.tfidf_index, 'get_stats') else {}
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the service"""
        self.logger.info("🛑 Shutting down RAG Service...")
        log_system_stop('rag_service', reason='shutdown')
        self.logger.info("✅ RAG Service shutdown complete")
