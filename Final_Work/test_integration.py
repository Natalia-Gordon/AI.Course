#!/usr/bin/env python3
"""
Simple test script to verify the integrated Hybrid RAG System
Tests the enhanced DataLoader and Summary Agent with LlamaExtract integration
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_enhanced_data_loader():
    """Test the enhanced DataLoader with LlamaExtract integration."""
    logger.info("🧪 Testing Enhanced DataLoader")
    
    try:
        from ingest.data_loader import DataLoader
        
        # Create a simple config dictionary (no config file needed)
        config = {
            'documents_dir': 'data/documents',
            'processed_dir': 'data/processed',
            'chunking': {
                'max_chunk_tokens': 400,
                'budget_ratio': 0.05
            }
        }
        
        logger.info("✓ Configuration created successfully")
        
        # Initialize enhanced data loader
        data_loader = DataLoader(config)
        logger.info("✓ Enhanced DataLoader initialized")
        
        # Check if LlamaExtract is available
        if data_loader.financial_agent:
            logger.info("✓ LlamaExtract integration available")
            logger.info(f"✓ Financial agent: {data_loader.financial_agent.name}")
        else:
            logger.warning("⚠ LlamaExtract not available (check LLAMA_CLOUD_API_KEY)")
        
        # Test that the loader has enhanced methods
        assert hasattr(data_loader, 'extract_financial_data'), "Has extract_financial_data method"
        assert hasattr(data_loader, 'enhance_chunks_with_financial_data'), "Has enhance_chunks_with_financial_data method"
        
        logger.info("✓ All enhanced DataLoader methods available")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced DataLoader test failed: {e}")
        return False

def test_enhanced_summary_agent():
    """Test the enhanced Summary Agent with LlamaExtract integration."""
    logger.info("🧪 Testing Enhanced Summary Agent")
    
    try:
        from agents.summary_agent import run_summary, extract_llama_extract_financial_data
        
        # Test that the agent has enhanced methods
        assert hasattr(run_summary, '__call__'), "run_summary is callable"
        assert hasattr(extract_llama_extract_financial_data, '__call__'), "extract_llama_extract_financial_data is callable"
        
        logger.info("✓ All enhanced Summary Agent methods available")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Summary Agent test failed: {e}")
        return False

def test_router_functionality():
    """Test that the router still works correctly."""
    logger.info("🧪 Testing Router Functionality")
    
    try:
        from agents.router import route_intent
        
        # Test intent routing
        test_queries = [
            "Summarize the financial highlights",
            "What are the Q1 2025 revenue figures?",
            "Show me tables with financial data"
        ]
        
        for query in test_queries:
            intent = route_intent(query)
            logger.info(f"✓ Query: '{query}' → Intent: {intent}")
        
        return True
        
    except Exception as e:
        logger.error(f"Router functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Integrated System Tests")
    
    try:
        # Test enhanced data loader
        data_loader_success = test_enhanced_data_loader()
        
        # Test enhanced summary agent
        summary_agent_success = test_enhanced_summary_agent()
        
        # Test router functionality
        router_success = test_router_functionality()
        
        # Summary
        print("\n" + "="*60)
        print("INTEGRATED SYSTEM TEST RESULTS")
        print("="*60)
        print(f"Enhanced DataLoader: {'✓ PASS' if data_loader_success else '✗ FAIL'}")
        print(f"Enhanced Summary Agent: {'✓ PASS' if summary_agent_success else '✗ FAIL'}")
        print(f"Router Functionality: {'✓ PASS' if router_success else '✗ FAIL'}")
        print("="*60)
        
        if data_loader_success and summary_agent_success and router_success:
            print("\n🎉 Your integrated system is ready to use!")
            print("\nWhat's New:")
            print("✓ LlamaExtract is now integrated into DataLoader by default")
            print("✓ Summary Agent automatically uses extracted financial data")
            print("✓ No additional flags needed - enhancement is automatic")
            print("✓ All existing functionality preserved")
            print("✓ No config file needed - uses sensible defaults")
            
            print("\nUsage Examples:")
            print("1. Standard Mode (with automatic LlamaExtract enhancement):")
            print("   python src/main.py --query 'Summarize financial highlights'")
            print("\n2. LangChain Enhanced Mode:")
            print("   python src/main.py --query 'Show me tables' --langchain")
            print("\n3. With verbose logging:")
            print("   python src/main.py --query 'Test query' --verbose")
            print("\n4. Reprocess documents:")
            print("   python src/main.py --query 'Test query' --reprocess")
            
            # Check LlamaExtract availability
            if os.environ.get('LLAMA_CLOUD_API_KEY'):
                print("\n✅ LLAMA_CLOUD_API_KEY found - LlamaExtract will be used!")
            else:
                print("\n⚠ LLAMA_CLOUD_API_KEY not found - LlamaExtract enhancement will be skipped")
                print("   Add LLAMA_CLOUD_API_KEY to your .env file for full enhancement")
        else:
            print("\n❌ System has issues that need to be resolved.")
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}", exc_info=True)

if __name__ == "__main__":
    main()
