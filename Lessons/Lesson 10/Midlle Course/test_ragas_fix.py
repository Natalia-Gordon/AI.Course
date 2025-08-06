#!/usr/bin/env python3
"""
Test script for the fixed RAGAS evaluation function
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function to test
from tool_functions import ragas_evaluate_summary_json

def test_ragas_parsing():
    """Test the RAGAS evaluation function with sample data"""
    
    # Test input (simplified version of what the agent sends)
    test_input = '''```
{
  "question": "What events occurred during the house robbery incident at 45 Elm Street?",
  "ground_truth": "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. Items stolen included electronics, jewelry, and cash totaling $15,700. The incident was discovered the next day and reported to police and insurance on April 14–15.",
  "summary": "A robbery occurred at Jennifer Lawson's residence on April 13, 2024, while she was away on a weekend trip. The house was unoccupied during the incident, and evidence of forced entry was found. Several high-value items were reported stolen, leading to the initiation of an insurance claim.",
  "contexts": [
    "A robbery occurred at Jennifer Lawson's residence on April 13, 2024, while she was away on a weekend trip.",
    "The house was unoccupied during the incident, and evidence of forced entry was found.",
    "Several high-value items were reported stolen, leading to the initiation of an insurance claim."
  ]
}
```'''
    
    print("Testing RAGAS evaluation function...")
    print("=" * 50)
    
    try:
        result = ragas_evaluate_summary_json(test_input)
        print(f"✅ SUCCESS: {result}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_ragas_parsing() 