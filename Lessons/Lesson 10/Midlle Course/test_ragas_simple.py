#!/usr/bin/env python3
"""
Simple test for RAGAS answer_correctness
"""

from ragas.metrics import answer_correctness
from datasets import Dataset

def test_ragas_simple():
    """Test RAGAS answer_correctness with simple data"""
    
    # Create simple test dataset
    dataset = Dataset.from_dict({
        "question": ["What happened at 45 Elm Street?"],
        "ground_truth": ["A robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away."],
        "answer": ["A robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away."],
        "contexts": [["The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away."]]
    })
    
    print("Testing RAGAS answer_correctness...")
    print(f"Dataset: {dataset}")
    
    try:
        # Use the score method
        result = answer_correctness.score(dataset)
        print(f"RAGAS Result: {result}")
        print(f"Score: {result.get('answer_correctness', 'N/A')}")
    except Exception as e:
        print(f"RAGAS Error: {e}")
        
        # Try alternative method
        try:
            result = answer_correctness.compute(dataset)
            print(f"RAGAS Compute Result: {result}")
        except Exception as e2:
            print(f"RAGAS Compute Error: {e2}")

if __name__ == "__main__":
    test_ragas_simple() 