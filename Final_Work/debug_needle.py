#!/usr/bin/env python3
"""
Debug script for Needle Agent text processing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agents.needle_agent import extract_relevant_snippet, calculate_relevance_score

def debug_text_processing():
    """Debug the text processing in Needle Agent."""
    
    # Sample text
    sample_text = "The company reported revenue of $15.2 million for Q1 2025, representing a 12% increase from the previous quarter. This growth was driven by strong performance in the insurance segment."
    
    # Query terms
    query_terms = ["revenue", "q1", "2025"]
    
    print("🔍 Debugging Needle Agent Text Processing")
    print("=" * 50)
    
    print(f"📝 Original text length: {len(sample_text)}")
    print(f"📝 Original text: {sample_text}")
    print(f"🔍 Query terms: {query_terms}")
    
    # Test relevance scoring
    score = calculate_relevance_score(query_terms, sample_text, {"section_type": "Financial"})
    print(f"📊 Relevance score: {score}")
    
    # Test snippet extraction
    snippet = extract_relevant_snippet(query_terms, sample_text, max_length=400)
    print(f"✂️  Extracted snippet length: {len(snippet)}")
    print(f"✂️  Extracted snippet: {snippet}")
    
    # Test with shorter max_length
    snippet_short = extract_relevant_snippet(query_terms, sample_text, max_length=50)
    print(f"✂️  Short snippet (max 50): {snippet_short}")
    
    # Test with medium max_length
    snippet_medium = extract_relevant_snippet(query_terms, sample_text, max_length=100)
    print(f"✂️  Medium snippet (max 100): {snippet_medium}")

if __name__ == "__main__":
    debug_text_processing()
