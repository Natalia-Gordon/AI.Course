#!/usr/bin/env python3
"""
Debug script to examine chunk structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import load_config
from ingest.data_loader import DataLoader

def debug_chunks():
    """Debug the structure of real chunks."""
    
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
        
        # Examine first few chunks
        print(f"\n🔍 Examining first 3 chunks:")
        print("-" * 50)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n📋 Chunk {i+1}:")
            print(f"   ID: {chunk.get('id', 'N/A')}")
            print(f"   File: {chunk.get('file_name', 'N/A')}")
            print(f"   Page: {chunk.get('page_number', 'N/A')}")
            print(f"   Section: {chunk.get('section_type', 'N/A')}")
            print(f"   Text length: {len(chunk.get('text', ''))}")
            print(f"   Text preview: {chunk.get('text', '')[:100]}...")
            print(f"   Keywords: {chunk.get('keywords', [])[:5]}")
            print(f"   Client ID: {chunk.get('client_id', 'N/A')}")
            print(f"   Case ID: {chunk.get('case_id', 'N/A')}")
            
            # Check for extracted financial data
            if 'extracted_financial_data' in chunk:
                print(f"   Financial data: {chunk['extracted_financial_data']}")
        
        # Search for specific terms
        print(f"\n🔍 Searching for specific terms in chunks:")
        print("-" * 50)
        
        search_terms = ["revenue", "net income", "customer deposits", "branch network"]
        
        for term in search_terms:
            found_chunks = []
            for chunk in chunks:
                if term.lower() in chunk.get('text', '').lower():
                    found_chunks.append({
                        'id': chunk.get('id'),
                        'page': chunk.get('page_number'),
                        'section': chunk.get('section_type'),
                        'text_preview': chunk.get('text', '')[:100]
                    })
            
            print(f"\n📝 Term '{term}': Found in {len(found_chunks)} chunks")
            for chunk_info in found_chunks[:3]:  # Show first 3
                print(f"   - {chunk_info['id']} (p{chunk_info['page']}, {chunk_info['section']}): {chunk_info['text_preview']}...")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chunks()
