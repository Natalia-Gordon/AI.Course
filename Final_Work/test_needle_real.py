#!/usr/bin/env python3
"""
Test Needle Agent with real data from the system using Hybrid Retrieval
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import load_config, build_indices
from ingest.data_loader import DataLoader
from agents.needle_agent import run_needle_with_hybrid_retrieval
from agents.router import route_intent
from retrieve.hybrid import HybridRetriever

def test_needle_with_hybrid_retrieval():
    """Test Needle Agent with hybrid retrieval (Pinecone + TF-IDF)."""
    
    print("🧪 Testing Needle Agent with Hybrid Retrieval (Pinecone + TF-IDF)")
    print("=" * 60)
    
    try:
        # Load configuration
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Initialize data loader
        data_loader = DataLoader(cfg)
        print("✅ DataLoader initialized")
        
        # Load existing chunks
        chunks = data_loader.get_processed_chunks('ayalon_q1_2025.pdf')
        print(f"✅ Loaded {len(chunks)} chunks from ayalon_q1_2025.pdf")
        
        if not chunks:
            print("❌ No chunks found")
            return
        
        # Build indices (including Pinecone)
        print("🔨 Building indices...")
        dense_index, sparse_index = build_indices(cfg, chunks)
        print("✅ Indices built successfully")
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            dense_index,
            sparse_index,
            namespace='ayalon_q1_2025'  # Use the namespace from your data
        )
        print("✅ Hybrid retriever created")
        
        # Test queries for Needle Agent
        test_queries = [
            "What was the revenue in Q1 2025?",
            "How much net income did the company report?",
            "Find information about customer deposits",
            "What is the page number for financial highlights?",
            "Show me the executive summary",
            "What are the key performance indicators?",
            "Find information about branch network expansion"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries with Needle Agent + Hybrid Retrieval")
        print("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 30)
            
            # Check intent routing
            intent = route_intent(query)
            print(f"🎯 Detected Intent: {intent}")
            
            if intent == "needle":
                try:
                    # Run Needle Agent with hybrid retrieval
                    answer = run_needle_with_hybrid_retrieval(query, hybrid_retriever, k=8)
                    print(f"✅ Needle Agent Answer (Hybrid Retrieval):\n{answer}")
                except Exception as e:
                    print(f"❌ Needle Agent Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Query routed to {intent} agent (not Needle)")
            
            print()
        
        print("🎯 Needle Agent testing with hybrid retrieval completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_needle_with_hybrid_retrieval()
