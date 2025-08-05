#!/usr/bin/env python3
"""
Test script for RAGAS evaluation
"""

import tool_functions

def test_ragas_evaluation():
    """Test the RAGAS evaluation function with simple data"""
    
    # Test data
    question = "What events occurred during the house robbery incident at 45 Elm Street?"
    ground_truth = "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. Items stolen included electronics, jewelry, and cash totaling $15,700. The incident was discovered the next day and reported to police and insurance on April 14–15."
    summary = "A robbery occurred at Jennifer Lawson's residence on April 13, 2024, while she was away on a weekend trip. The house was forcibly entered, and several high-value items were reported stolen. The estimated time of robbery was between 8:00 PM and 10:00 PM. The total estimated loss was $15,700."
    contexts = [
        "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away.",
        "The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor.",
        "Items stolen included electronics, jewelry, and cash totaling $15,700."
    ]
    
    print("Testing RAGAS evaluation...")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Summary: {summary}")
    print(f"Contexts: {contexts}")
    print("-" * 50)
    
    # Test the function directly
    result = tool_functions.ragas_evaluate_summary(
        question=question,
        ground_truth=ground_truth,
        summary=summary,
        contexts=contexts
    )
    
    print(f"Result: {result}")
    
    # Test the JSON wrapper
    test_json = {
        "question": question,
        "ground_truth": ground_truth,
        "summary": summary,
        "contexts": contexts
    }
    
    json_str = json.dumps(test_json, indent=2)
    print("\nTesting JSON wrapper...")
    result2 = tool_functions.ragas_evaluate_summary_json(json_str)
    print(f"JSON Result: {result2}")

if __name__ == "__main__":
    import json
    test_ragas_evaluation() 