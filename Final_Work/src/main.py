#!/usr/bin/env python3
"""
Main Entry Point for the Hybrid RAG System
Now with integrated LlamaExtract enhancement in the DataLoader
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (same as Middle Course)
load_dotenv()

# Import existing modules
from index.pinecone_index import PineconeIndex
from index.tfidf_index import TfidfIndexer as TFIDFIndex
from retrieve.hybrid import HybridRetriever
from retrieve.rerank import Reranker
from agents.router import route_intent

# Optional LangChain enhancement
try:
    from agents.langchain_agents import LangChainEnhancedAgents
    LANGCHAIN_AVAILABLE = True
    logger.info("✓ LangChain enhancement available")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.info("⚠ LangChain enhancement not available, using standard agents")

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / 'config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from: {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        logger.error("Configuration file is required. Please create src/config.yaml")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing config file: {e}")
        logger.error("Please check your YAML syntax in src/config.yaml")
        raise yaml.YAMLError(f"Invalid YAML in config file: {e}")

def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    required_vars = ['PINECONE_API_KEY', 'PINECONE_REGION', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file")
        return False
    
    logger.info("✓ All required environment variables are set")
    return True

def build_indices(config: Dict[str, Any], chunks: List[Any]) -> tuple[Optional[PineconeIndex], TFIDFIndex]:
    """Build dense and sparse indices."""
    logger.info("Building indices...")
    
    # Build sparse index
    sparse_index = TFIDFIndex()
    sparse_index.fit(chunks)
    logger.info("✓ Sparse index built successfully")
    
    # Build dense index (Pinecone)
    dense_index = None
    try:
        dense_index = PineconeIndex(
            api_key=os.environ.get('PINECONE_API_KEY'),
            region=os.environ.get('PINECONE_REGION'),
            cloud=os.environ.get('PINECONE_CLOUD'),
            index_name=config['pinecone']['index_name'],
            dimension=config['embedding']['dim'],
            metric=os.environ.get('PINECONE_METRIC', 'cosine')
        )
        # Extract namespace for Pinecone indexing
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0]
            if isinstance(first_chunk, dict):
                file_name = first_chunk.get('file_name', 'Unknown')
            else:
                file_name = getattr(first_chunk, 'file_name', 'Unknown')
            
            # Create meaningful namespace from filename
            if 'ayalon' in file_name.lower():
                pinecone_namespace = 'ayalon_q1_2025'
            elif 'financial' in file_name.lower():
                pinecone_namespace = 'financial_reports'
            else:
                pinecone_namespace = 'general_docs'
            
            logger.info(f"Using Pinecone namespace: {pinecone_namespace}")
        else:
            pinecone_namespace = 'default'
            logger.info("Using default Pinecone namespace")
        
        # Check if we need to recreate with new namespace
        if dense_index.check_index_content():
            # Check current namespace
            stats = dense_index.get_index_stats()
            if stats and stats.namespaces:
                current_namespace = list(stats.namespaces.keys())[0]
                if current_namespace == "__default__" and pinecone_namespace != "__default__":
                    logger.info(f"🔄 Recreating index to move from {current_namespace} to {pinecone_namespace}")
                    dense_index.recreate_with_namespace(chunks, pinecone_namespace)
                else:
                    logger.info(f"✅ Using existing namespace: {current_namespace}")
            else:
                dense_index.build_index(chunks, namespace=pinecone_namespace)
        else:
            # Index is empty, build normally
            dense_index.build_index(chunks, namespace=pinecone_namespace)
        
        logger.info("✓ Dense index built successfully")
        
        # Verify that vectors were actually inserted
        if dense_index.check_index_content():
            logger.info("✅ Pinecone index verification successful")
        else:
            logger.warning("⚠ Pinecone index verification failed - index may be empty")
    except Exception as e:
        logger.warning(f"Pinecone index failed: {e}")
        logger.info("Continuing with sparse retrieval only")
    
    return dense_index, sparse_index

def process_query(query: str, chunks: List[Any], config: Dict[str, Any], use_langchain: bool = False) -> Dict[str, Any]:
    """Process a query through the RAG pipeline with optional LangChain enhancement."""
    logger.info(f"Processing query: {query}")
    
    # Build indices
    dense_index, sparse_index = build_indices(config, chunks)
    
    # Extract namespace from document metadata (use first chunk's client_id or default)
    namespace = None
    if chunks and len(chunks) > 0:
        first_chunk = chunks[0]
        logger.info(f"Analyzing first chunk for namespace: {type(first_chunk)}")
        
        if isinstance(first_chunk, dict):
            chunk_id = first_chunk.get('id')
            client_id = first_chunk.get('client_id')
            case_id = first_chunk.get('case_id')
            file_name = first_chunk.get('file_name', 'Unknown')
            logger.info(f"Dictionary chunk - ID: {chunk_id}, Client ID: {client_id}, Case ID: {case_id}, File: {file_name}")
        else:
            chunk_id = getattr(first_chunk, 'id', None)
            client_id = getattr(first_chunk, 'client_id', None)
            case_id = getattr(first_chunk, 'case_id', None)
            file_name = getattr(first_chunk, 'file_name', 'Unknown')
            logger.info(f"Object chunk - ID: {chunk_id}, Client ID: {client_id}, Case ID: {case_id}, File: {file_name}")
        
        # For financial reports, use filename-based namespace (case_id is not applicable)
        if 'ayalon' in file_name.lower():
            namespace = 'ayalon_q1_2025'
        elif 'financial' in file_name.lower():
            namespace = 'financial_reports'
        else:
            namespace = 'general_docs'
        
        logger.info(f"📋 Document identifiers:")
        logger.info(f"   Client ID: {client_id} (company identifier)")
        logger.info(f"   Case ID: {case_id} (not applicable for financial reports)")
        logger.info(f"   File: {file_name}")
        logger.info(f"   Final Namespace: {namespace}")
    
    # Use default namespace if none found
    if not namespace:
        namespace = 'default'
        logger.info("Using default namespace")
    
    logger.info(f"Final namespace selected: {namespace}")
    
    # Create retriever
    retriever = HybridRetriever(
        dense_index,
        sparse_index,
        Reranker(),
        namespace=namespace
    )
    
    # Search for relevant chunks
    logger.info("Searching for relevant chunks...")
    hits = retriever.search(
        query,
        k_dense=config['retrieval']['dense_k'],
        k_sparse=config['retrieval']['sparse_k'],
        final_k=config['retrieval']['final_k']
    )
    
    logger.info(f"Retrieved {len(hits)} relevant chunks")
    
    # Route to appropriate agent (KEEPING YOUR CURRENT ARCHITECTURE)
    intent = route_intent(query)
    logger.info(f"Detected intent: {intent}")
    
    # Generate answer based on intent with optional LangChain enhancement
    if use_langchain and LANGCHAIN_AVAILABLE:
        try:
            # Initialize LangChain enhanced agents
            langchain_agents = LangChainEnhancedAgents(config)
            
            # Set the retriever for hybrid search functionality
            langchain_agents.set_retriever(retriever)
            
            # Process with LangChain
            answer = langchain_agents.process_with_langchain(query, intent, hits)
            logger.info("✓ Query processed with LangChain enhancement")
            
        except Exception as e:
            logger.warning(f"LangChain processing failed: {e}, falling back to standard agents")
            answer = _fallback_to_standard_agents(query, intent, hits)
    else:
        # Use standard agents (your current implementation)
        answer = _fallback_to_standard_agents(query, intent, hits)
    
    return {
        'query': query,
        'intent': intent,
        'hits': hits,
        'answer': answer,
        'processing_method': 'langchain' if (use_langchain and LANGCHAIN_AVAILABLE) else 'standard'
    }

def _fallback_to_standard_agents(query: str, intent: str, hits: List[Dict]) -> str:
    """Fallback to standard agent functions."""
    logger.info(f"Using standard agent processing for intent: {intent}")
    
    if intent == 'summary':
        from agents.summary_agent import run_summary
        return run_summary(query, hits)
    elif intent == 'needle':
        from agents.needle_agent import run_needle
        return run_needle(query, hits)
    else:  # table
        from agents.table_qa_agent import run_table_qa
        return run_table_qa(query, hits)

def display_results(results: Dict[str, Any]) -> None:
    """Display query results in a formatted way."""
    query = results['query']
    intent = results['intent']
    hits = results['hits']
    answer = results['answer']
    method = results.get('processing_method', 'unknown')
    
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    print(f"Query: {query}")
    print(f"Intent: {intent}")
    print(f"Processing Method: {method.upper()}")
    print(f"Retrieved chunks: {len(hits)}")
    
    print("\nTOP CHUNKS:")
    for i, hit in enumerate(hits[:3]):
        summary = hit.get('chunk_summary', hit.get('text', '')[:100])
        print(f"{i+1}. {summary}...")
        print(f"   Source: {hit.get('file_name')} | Page: {hit.get('page_number')} | Type: {hit.get('section_type')}")
        
        # Show extracted data if available
        if hit.get('extracted_financial_data'):
            financial_data = hit['extracted_financial_data']
            print(f"   📊 Enhanced with extracted financial data:")
            if financial_data.get('company_name'):
                print(f"      Company: {financial_data['company_name']}")
            if financial_data.get('revenue'):
                print(f"      Revenue: {financial_data['revenue']}")
            if financial_data.get('kpis'):
                print(f"      KPIs: {', '.join(financial_data['kpis'][:3])}")
        print()
    
    print("\nANSWER:")
    print(answer)
    print("="*60)

def main():
    """Main function."""
    logger.info("🚀 Starting Enhanced Hybrid RAG System")
    logger.info("✨ LlamaExtract enhancement integrated into DataLoader")
    if LANGCHAIN_AVAILABLE:
        logger.info("✨ LangChain enhancement available")
    
    # Parse arguments
    ap = argparse.ArgumentParser(description='Enhanced Hybrid RAG System for Financial Documents')
    ap.add_argument('--query', required=True, help='Query to process')
    ap.add_argument('--document', default='ayalon_q1_2025.pdf', help='Document to process')
    ap.add_argument('--reprocess', action='store_true', help='Reprocess documents even if chunks exist')
    ap.add_argument('--langchain', action='store_true', help='Enable LangChain enhancement (if available)')
    ap.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = ap.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)
        
        # Load configuration from config.yaml file
        try:
            cfg = load_config()
            logger.info(f"Configuration loaded: {cfg['project_name']}")
        except (FileNotFoundError, yaml.YAMLError) as config_error:
            logger.error(f"Configuration error: {config_error}")
            logger.error("Please ensure src/config.yaml exists and has valid YAML syntax")
            sys.exit(1)
        
        # Initialize enhanced data loader (now includes LlamaExtract by default)
        from ingest.data_loader import DataLoader
        data_loader = DataLoader(cfg)
        logger.info("✓ Enhanced DataLoader initialized with LlamaExtract integration")
        
        # Show LlamaExtract status
        llama_status = data_loader.get_llama_extract_status()
        logger.info(f"LlamaExtract Status: {llama_status}")
        
        if data_loader.is_llama_extract_available():
            logger.info("🎉 LlamaExtract enhancement will be used for document processing")
        else:
            logger.info("⚠ LlamaExtract enhancement not available, using standard processing")
        
        # Get or create chunks
        chunks = []
        if not args.reprocess:
            logger.info("Checking for existing chunks...")
            chunks = data_loader.get_processed_chunks(args.document)
            if chunks:
                logger.info(f"Loaded {len(chunks)} existing chunks for {args.document}")
        
        if not chunks:
            logger.info(f"Processing document: {args.document}")
            chunks = data_loader.load_document(args.document)
            data_loader.save_chunks(chunks, args.document)
            logger.info(f"✓ Document processed and saved: {len(chunks)} chunks created")
        
        # Process query
        use_langchain = args.langchain and LANGCHAIN_AVAILABLE
        
        if args.langchain and not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain requested but not available, using standard agents")
        
        logger.info(f"Processing query with LangChain: {'enabled' if use_langchain else 'disabled'}")
        
        results = process_query(args.query, chunks, cfg, use_langchain)
        
        # Display results
        display_results(results)
        
        logger.info("Query processing completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
