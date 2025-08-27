import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    context_relevancy
)

class RAGASEvaluator:
    """RAGAS-based evaluation for the Hybrid RAG system."""
    
    def __init__(self, test_data_path: str = "data/ragas_testset.json"):
        self.test_data_path = Path(test_data_path)
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from JSON file."""
        if not self.test_data_path.exists():
            return []
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_evaluation_dataset(self, questions: List[Dict], contexts: List[Dict], 
                                 answers: List[str], ground_truth: List[Dict]) -> pd.DataFrame:
        """Prepare dataset for RAGAS evaluation."""
        if len(questions) != len(contexts) != len(answers) != len(ground_truth):
            raise ValueError("All input lists must have the same length")
        
        dataset = []
        for i in range(len(questions)):
            dataset.append({
                'question': questions[i],
                'contexts': contexts[i],
                'answer': answers[i],
                'ground_truth': ground_truth[i]
            })
        
        return pd.DataFrame(dataset)
    
    def evaluate_system(self, dataset: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the system using RAGAS metrics."""
        if dataset.empty:
            return {}
        
        # Run RAGAS evaluation
        results = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                context_relevancy
            ]
        )
        
        # Extract metric values
        metrics = {}
        for metric_name, metric_value in results.items():
            if hasattr(metric_value, 'value'):
                metrics[metric_name] = float(metric_value.value)
            else:
                metrics[metric_name] = float(metric_value)
        
        return metrics
    
    def check_requirements(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Check if metrics meet project requirements."""
        requirements = {
            'context_precision': {'target': 0.75, 'current': metrics.get('context_precision', 0)},
            'context_recall': {'target': 0.70, 'current': metrics.get('context_recall', 0)},
            'faithfulness': {'target': 0.85, 'current': metrics.get('faithfulness', 0)},
            'answer_relevancy': {'target': 0.80, 'current': metrics.get('answer_relevancy', 0)},
            'context_relevancy': {'target': 0.80, 'current': metrics.get('context_relevancy', 0)}
        }
        
        # Add status for each metric
        for metric_name, metric_data in requirements.items():
            current = metric_data['current']
            target = metric_data['target']
            metric_data['status'] = 'PASS' if current >= target else 'FAIL'
            metric_data['gap'] = target - current
        
        return requirements
    
    def generate_evaluation_report(self, metrics: Dict[str, float], 
                                 requirements: Dict[str, Dict[str, Any]]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("RAGAS EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall performance
        passed_metrics = sum(1 for req in requirements.values() if req['status'] == 'PASS')
        total_metrics = len(requirements)
        overall_score = passed_metrics / total_metrics
        
        report.append(f"OVERALL PERFORMANCE: {overall_score:.1%} ({passed_metrics}/{total_metrics} metrics passed)")
        report.append("")
        
        # Individual metrics
        report.append("METRIC DETAILS:")
        report.append("-" * 40)
        
        for metric_name, metric_data in requirements.items():
            status_icon = "✅" if metric_data['status'] == 'PASS' else "❌"
            report.append(f"{status_icon} {metric_name.upper()}")
            report.append(f"   Current: {metric_data['current']:.3f}")
            report.append(f"   Target:  {metric_data['target']:.3f}")
            report.append(f"   Status:  {metric_data['status']}")
            if metric_data['status'] == 'FAIL':
                report.append(f"   Gap:     {metric_data['gap']:.3f}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        failed_metrics = [name for name, data in requirements.items() if data['status'] == 'FAIL']
        
        if not failed_metrics:
            report.append("🎉 All metrics meet requirements! System is performing excellently.")
        else:
            for metric in failed_metrics:
                if metric == 'context_precision':
                    report.append("• Improve context precision by enhancing retrieval relevance")
                elif metric == 'context_recall':
                    report.append("• Improve context recall by retrieving more comprehensive information")
                elif metric == 'faithfulness':
                    report.append("• Improve faithfulness by ensuring answers are grounded in retrieved contexts")
                elif metric == 'answer_relevancy':
                    report.append("• Improve answer relevancy by enhancing answer generation quality")
                elif metric == 'context_relevancy':
                    report.append("• Improve context relevancy by better query understanding")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, metrics: Dict[str, float], 
                               requirements: Dict[str, Dict[str, Any]], 
                               output_path: str = "evaluation_results.json"):
        """Save evaluation results to file."""
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'requirements_check': requirements,
            'summary': {
                'total_metrics': len(requirements),
                'passed_metrics': sum(1 for req in requirements.values() if req['status'] == 'PASS'),
                'overall_score': sum(1 for req in requirements.values() if req['status'] == 'PASS') / len(requirements)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to {output_path}")
    
    def run_full_evaluation(self, questions: List[str], contexts: List[List[str]], 
                           answers: List[str], ground_truth: List[Dict]) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        print("Starting RAGAS evaluation...")
        
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(questions, contexts, answers, ground_truth)
        print(f"Prepared dataset with {len(dataset)} samples")
        
        # Run evaluation
        metrics = self.evaluate_system(dataset)
        print("Evaluation completed")
        
        # Check requirements
        requirements = self.check_requirements(metrics)
        
        # Generate report
        report = self.generate_evaluation_report(metrics, requirements)
        print("\n" + report)
        
        # Save results
        self.save_evaluation_results(metrics, requirements)
        
        return {
            'metrics': metrics,
            'requirements': requirements,
            'report': report,
            'dataset_size': len(dataset)
        }
