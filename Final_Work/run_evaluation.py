#!/usr/bin/env python3
"""
Comprehensive evaluation script for the Hybrid RAG System
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from eval.ragas_evaluator import RAGASEvaluator
from ingest.data_loader import DataLoader
from pipeline.hybrid_rag_pipeline import HybridRAGPipeline
import yaml

def load_evaluation_data():
    """Load evaluation questions and ground truth."""
    data_dir = Path('data')
    
    # Load evaluation questions
    questions_file = data_dir / 'evaluation_questions.json'
    if questions_file.exists():
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    else:
        print("Warning: evaluation_questions.json not found")
        questions_data = []
    
    # Load ground truth
    ground_truth_file = data_dir / 'ground_truth.json'
    if ground_truth_file.exists():
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
    else:
        print("Warning: ground_truth.json not found")
        ground_truth_data = []
    
    # Load RAGAS test set
    ragas_file = data_dir / 'ragas_testset.json'
    if ragas_file.exists():
        with open(ragas_file, 'r', encoding='utf-8') as f:
            ragas_data = json.load(f)
    else:
        print("Warning: ragas_testset.json not found")
        ragas_data = []
    
    return questions_data, ground_truth_data, ragas_data

def run_hybrid_rag_queries(pipeline: HybridRAGPipeline, questions: List[str]) -> List[Dict[str, Any]]:
    """Run queries through the Hybrid RAG pipeline."""
    results = []
    
    print(f"Running {len(questions)} queries through Hybrid RAG pipeline...")
    
    for i, question in enumerate(questions):
        print(f"Processing query {i+1}/{len(questions)}: {question[:50]}...")
        
        try:
            response = pipeline.query(question)
            results.append({
                'question': question,
                'answer': response.get('answer', ''),
                'contexts': response.get('contexts', []),
                'query_id': i
            })
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            results.append({
                'question': question,
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'query_id': i
            })
    
    return results

def prepare_ragas_dataset(questions: List[str], results: List[Dict], 
                         ground_truth: List[Dict]) -> tuple:
    """Prepare dataset for RAGAS evaluation."""
    # Extract questions, contexts, answers, and ground truth
    q_list = []
    c_list = []
    a_list = []
    gt_list = []
    
    for i, result in enumerate(results):
        if i < len(questions) and i < len(ground_truth):
            q_list.append(result['question'])
            
            # Extract context texts
            contexts = [ctx.get('text', '')[:500] for ctx in result.get('contexts', [])]
            c_list.append(contexts)
            
            a_list.append(result.get('answer', ''))
            gt_list.append(ground_truth[i])
    
    return q_list, c_list, a_list, gt_list

def main():
    """Main evaluation function."""
    print("=" * 60)
    print("HYBRID RAG SYSTEM EVALUATION")
    print("=" * 60)
    
    # Load configuration
    config_path = Path('src/config.yaml')
    if not config_path.exists():
        print("Error: config.yaml not found")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load evaluation data
    print("\n1. Loading evaluation data...")
    questions_data, ground_truth_data, ragas_data = load_evaluation_data()
    
    if not questions_data:
        print("No evaluation questions found. Creating sample questions...")
        questions_data = [
            "What is the revenue for Q1 2025?",
            "Summarize the financial highlights",
            "What are the key risk factors mentioned?",
            "Find information about the company's performance",
            "What tables are available in the document?"
        ]
    
    # Initialize pipeline
    print("\n2. Initializing Hybrid RAG pipeline...")
    try:
        pipeline = HybridRAGPipeline(str(config_path))
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Continuing with evaluation using available data...")
        pipeline = None
    
    # Run queries if pipeline is available
    if pipeline:
        print("\n3. Running queries through Hybrid RAG pipeline...")
        results = run_hybrid_rag_queries(pipeline, questions_data)
    else:
        print("\n3. Skipping query execution (pipeline not available)")
        results = []
    
    # Prepare RAGAS dataset
    print("\n4. Preparing RAGAS evaluation dataset...")
    if results and ground_truth_data:
        q_list, c_list, a_list, gt_list = prepare_ragas_dataset(
            questions_data, results, ground_truth_data
        )
        
        if len(q_list) > 0:
            print(f"Prepared dataset with {len(q_list)} samples")
            
            # Run RAGAS evaluation
            print("\n5. Running RAGAS evaluation...")
            evaluator = RAGASEvaluator()
            
            try:
                evaluation_results = evaluator.run_full_evaluation(
                    q_list, c_list, a_list, gt_list
                )
                
                print("\n6. Evaluation completed successfully!")
                print(f"Dataset size: {evaluation_results['dataset_size']}")
                
            except Exception as e:
                print(f"Error during RAGAS evaluation: {e}")
                print("This might be due to missing dependencies or data format issues")
        else:
            print("No valid data for evaluation")
    else:
        print("No results or ground truth available for evaluation")
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"✓ Queries processed: {len(results)}")
        print(f"✓ Pipeline status: {'Available' if pipeline else 'Not available'}")
        
        # Check answer quality
        valid_answers = [r for r in results if r.get('answer') and not r['answer'].startswith('Error')]
        print(f"✓ Valid answers generated: {len(valid_answers)}/{len(results)}")
        
        if valid_answers:
            avg_contexts = sum(len(r.get('contexts', [])) for r in valid_answers) / len(valid_answers)
            print(f"✓ Average contexts per answer: {avg_contexts:.1f}")
    else:
        print("✗ No results generated")
    
    print("\nEvaluation completed. Check evaluation_results.json for detailed metrics.")

if __name__ == "__main__":
    main()
