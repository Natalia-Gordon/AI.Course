#!/usr/bin/env python3
"""
Display RAGAS Evaluation Results in Table Format
"""

import json

def show_results():
    # Load results
    with open('data/eval/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print("🎯 RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Targets Met: {results['met_targets']}/4")
    print()
    
    print("📊 METRICS PERFORMANCE TABLE:")
    print("=" * 60)
    print(f"{'Metric':<20} {'Current':<10} {'Target':<8} {'Status':<8} {'Performance':<12}")
    print("-" * 60)
    
    for metric, data in results['requirements'].items():
        status_icon = "✅" if data['status'] == 'PASS' else "❌"
        metric_name = metric.replace("_", " ").title()
        current = data['current']
        target = data['target']
        status = data['status']
        
        # Calculate performance percentage
        if target > 0:
            performance_pct = (current / target) * 100
            performance = f"{performance_pct:.1f}%"
        else:
            performance = "N/A"
        
        print(f"{metric_name:<20} {current:<10.3f} {target:<8.2f} {status_icon} {status:<6} {performance:<12}")
    
    print()
    print("🔍 DETAILED TEST CASE METRICS:")
    print("=" * 80)
    print(f"{'Case':<4} {'Question':<35} {'Faith':<6} {'Relevancy':<10} {'Precision':<10} {'Recall':<7}")
    print("-" * 80)
    
    # Check if we have individual test case results
    test_case_results = None
    if 'test_case_results' in results:
        test_case_results = results['test_case_results']
    elif 'average_metrics' in results and 'test_case_results' in results['average_metrics']:
        test_case_results = results['average_metrics']['test_case_results']
    
    if test_case_results:
        for case_result in test_case_results:
            case_num = case_result['case_number']
            question = case_result['question'][:34] + "..." if len(case_result['question']) > 34 else case_result['question']
            
            # Extract metrics for this case
            metrics = case_result['metrics']
            faithfulness = metrics.get('faithfulness', 0.0)
            answer_relevancy = metrics.get('answer_relevancy', 0.0)
            context_precision = metrics.get('context_precision', 0.0)
            context_recall = metrics.get('context_recall', 0.0)
            
            print(f"{case_num:<4} {question:<35} {faithfulness:<6.3f} {answer_relevancy:<10.3f} {context_precision:<10.3f} {context_recall:<7.3f}")
    else:
        # Fallback to basic test case display
        for i, test_case in enumerate(results['test_cases'], 1):
            question = test_case['question'][:34] + "..." if len(test_case['question']) > 34 else test_case['question']
            print(f"{i:<4} {question:<35} {'N/A':<6} {'N/A':<10} {'N/A':<10} {'N/A':<7}")
    
    print()
    print("📈 SUMMARY STATISTICS:")
    print("-" * 30)
    print(f"• Total Test Cases: {results['total_cases']}")
    print(f"• Evaluation Strategy: {results['evaluation_strategy']}")
    
    # Calculate overall score safely
    try:
        if 'average_metrics' in results:
            metrics = results['average_metrics']
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'test_case_results'}
            if numeric_metrics:
                overall_score = sum(numeric_metrics.values()) / len(numeric_metrics)
                print(f"• Overall Score: {overall_score:.3f}")
            else:
                print("• Overall Score: N/A")
        else:
            print("• Overall Score: N/A")
    except Exception as e:
        print(f"• Overall Score: Error calculating ({e})")
    
    print()
    print("🎯 TARGET REQUIREMENTS:")
    print("-" * 30)
    print("• Context Precision ≥ 0.75")
    print("• Context Recall ≥ 0.70")
    print("• Faithfulness ≥ 0.85")
    print("• Answer Relevancy ≥ 0.80")

if __name__ == "__main__":
    show_results()
