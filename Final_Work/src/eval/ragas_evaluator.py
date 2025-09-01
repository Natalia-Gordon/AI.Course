#!/usr/bin/env python3
"""
Enhanced RAGAS Evaluation System for Hybrid RAG
Uses new architecture with proper logging, configuration, and component separation
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import proper Ragas metrics and components
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import our custom modules
from .config import EvaluationConfigManager
from .ground_truth_manager import GroundTruthManager
from .test_set_generator import TestSetGenerator
from .metrics_calculator import MetricsCalculator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger, log_system_start

logger = get_logger(__name__)

class RAGASEvaluator:
    """Enhanced RAGAS Evaluator using current Ragas API"""
    
    def __init__(self, config: EvaluationConfigManager):
        """Initialize RAGAS Evaluator with current Ragas API"""
        self.config = config
        self.ground_truth_manager = GroundTruthManager(config)
        self.test_set_generator = TestSetGenerator(config)
        self.metrics_calculator = MetricsCalculator(config)
        
        # Load data
        self.evaluation_questions = self._load_evaluation_questions()
        self.ground_truth = self._load_ground_truth()
        self.ragas_testset = self._load_ragas_testset()
        
        # Initialize proper Ragas metrics as per current API
        self._initialize_ragas_metrics()
        
        logger.info("🚀 Enhanced RAGAS Evaluator initialized")
        logger.info(f"📊 Target metrics: {self.config.get_target_metrics()}")
    
    def _initialize_ragas_metrics(self):
        """Initialize proper Ragas metrics using current API"""
        try:
            logger.info("🔧 Starting Ragas metrics initialization...")
            
            # Initialize OpenAI LLM for Ragas metrics
            logger.info("🔧 Creating OpenAI LLM wrapper...")
            openai_llm = LangchainLLMWrapper(
                langchain_llm=ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=self.config.get_openai_api_key()
                )
            )
            logger.info("✅ OpenAI LLM wrapper created successfully")
            
            # Initialize embeddings for metrics that need them
            logger.info("🔧 Creating embeddings model...")
            base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.config.get_openai_api_key())
            embeddings_model = LangchainEmbeddingsWrapper(embeddings=base_embeddings)
            logger.info("✅ Embeddings model created successfully")
            
            # Create Ragas metrics with LLM and embeddings
            logger.info("🔧 Creating faithfulness metric...")
            faithfulness_metric = faithfulness
            faithfulness_metric.llm = openai_llm
            logger.info("✅ Faithfulness metric created")
            
            logger.info("🔧 Creating answer_relevancy metric...")
            answer_relevancy_metric = answer_relevancy
            answer_relevancy_metric.llm = openai_llm
            answer_relevancy_metric.embeddings = embeddings_model
            logger.info("✅ Answer relevancy metric created")
            
            logger.info("🔧 Creating context_precision metric...")
            context_precision_metric = context_precision
            context_precision_metric.llm = openai_llm
            logger.info("✅ Context precision metric created")
            
            logger.info("🔧 Creating context_recall metric...")
            context_recall_metric = context_recall
            context_recall_metric.llm = openai_llm
            context_recall_metric.embeddings = embeddings_model
            logger.info("✅ Context recall metric created")
            
            self.ragas_metrics = {
                'faithfulness': faithfulness_metric,
                'answer_relevancy': answer_relevancy_metric,
                'context_precision': context_precision_metric,
                'context_recall': context_recall_metric
            }
            
            # Initialize run config
            self.run_config = RunConfig()
            
            # Initialize all metrics with run config
            for metric in self.ragas_metrics.values():
                metric.init(self.run_config)
            
            logger.info("✅ Ragas metrics initialized successfully")
            logger.info(f"🔧 Available metrics: {list(self.ragas_metrics.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ragas metrics: {e}")
            raise
    
    def _load_evaluation_questions(self) -> List[str]:
        """Load evaluation questions from file"""
        try:
            questions_file = Path('data/eval/evaluation_questions.json')
            if questions_file.exists():
                with open(questions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    questions = [item['question'] for item in data]
                    logger.info(f"✅ Loaded {len(questions)} evaluation questions")
                    return questions
            else:
                logger.warning("⚠️ evaluation_questions.json not found, using default questions")
                return self._get_default_questions()
        except Exception as e:
            logger.error(f"❌ Error loading evaluation questions: {e}")
            return self._get_default_questions()
    
    def _get_default_questions(self) -> List[str]:
        """Get default evaluation questions if file not found"""
        return [
            "What are the key financial highlights for Q1 2025?",
            "What operational improvements were mentioned?",
            "What are the main business segments?",
            "What financial metrics are shown in the tables?"
        ]
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth data"""
        try:
            return self.ground_truth_manager.load_ground_truth()
        except Exception as e:
            logger.error(f"❌ Error loading ground truth: {e}")
            return []
    
    def _load_ragas_testset(self) -> List[Dict[str, Any]]:
        """Load RAGAS test set"""
        try:
            testset_file = Path('data/eval/ragas_testset.json')
            if testset_file.exists():
                with open(testset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ Loaded {len(data)} RAGAS test cases")
                    return data
            else:
                logger.warning("⚠️ ragas_testset.json not found")
                return []
        except Exception as e:
            logger.error(f"❌ Error loading RAGAS test set: {e}")
            return []
    
    def run_complete_evaluation(self, testset: Optional[Dict] = None) -> Dict[str, Any]:
        """Run complete evaluation using current Ragas API"""
        try:
            start_time = time.time()
            
            # Use provided test set or fallback to original
            if testset:
                logger.info("🔧 Using provided enhanced test set...")
                testset_to_use = testset
                test_cases_count = len(testset)
            else:
                logger.info("🔧 Using original test set...")
                testset_to_use = self.ragas_testset
                test_cases_count = len(self.ragas_testset)
            
            log_system_start('ragas_evaluation', 
                            questions_count=len(self.evaluation_questions),
                            ground_truth_count=len(self.ground_truth),
                            test_cases_count=test_cases_count)
            
            logger.info("🚀 Starting Enhanced RAGAS Evaluation")
            logger.info("Using current Ragas API with proper integration")
            
            # Validate data
            logger.info(f"✅ Loaded {len(self.evaluation_questions)} evaluation questions")
            logger.info(f"✅ Loaded {len(self.ground_truth)} ground truth entries")
            logger.info(f"✅ Loaded {test_cases_count} test cases")
            
            # Validate ground truth data
            validation_results = self.ground_truth_manager.validate_ground_truth()
            if validation_results['invalid_entries'] > 0:
                logger.warning(f"⚠️ Found {validation_results['invalid_entries']} invalid ground truth entries")
            
            # Use selected test set
            logger.info(f"🔧 Using {'enhanced' if testset else 'original'} test set...")
            original_testset = testset_to_use
            
            # Validate original test set
            validation_results = self.test_set_generator.validate_test_set(original_testset)
            if validation_results['invalid_cases'] > 0:
                logger.warning(f"⚠️ Found {validation_results['invalid_cases']} invalid test cases")
            
            # Run RAGAS evaluation using current API
            logger.info("📊 Running RAGAS evaluation with current API...")
            evaluation_results = self._run_ragas_evaluation_with_current_api(original_testset)
            
            # Calculate final metrics
            final_metrics = self._calculate_final_metrics(evaluation_results)
            
            # Save results
            self._save_evaluation_results(final_metrics)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Complete evaluation finished successfully in {execution_time:.3f}s")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def _run_ragas_evaluation_with_current_api(self, testset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run RAGAS evaluation using current API with SingleTurnSample"""
        try:
            logger.info("🤖 Running RAGAS evaluation with current API")
            
            all_metrics = {
                'faithfulness': [],
                'answer_relevancy': [],
                'context_precision': [],
                'context_recall': []
            }
            
            # Store individual test case results
            test_case_results = []
            
            # Process each test case using current Ragas API
            for i, test_case in enumerate(testset):
                logger.info(f"🔍 Processing test case {i+1}/{len(testset)}: {test_case['question'][:50]}...")
                
                # Create SingleTurnSample as required by current Ragas API
                sample = SingleTurnSample(
                    user_input=test_case['question'],
                    retrieved_contexts=test_case['contexts'],
                    response=test_case['answer'],
                    reference=test_case['ground_truth']
                )
                
                # Evaluate each metric using current API
                case_metrics = {}
                for metric_name, metric in self.ragas_metrics.items():
                    try:
                        # Score the sample using the metric
                        score = metric.single_turn_score(sample)
                        
                        case_metrics[metric_name] = score
                        all_metrics[metric_name].append(score)
                        
                        logger.debug(f"✅ {metric_name}: {score}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to evaluate {metric_name}: {e}")
                        case_metrics[metric_name] = 0.0
                        all_metrics[metric_name].append(0.0)
                
                # Store individual test case result
                test_case_results.append({
                    'case_number': i + 1,
                    'question': test_case['question'],
                    'metrics': case_metrics
                })
                
                logger.info(f"📊 Case {i+1} metrics: {case_metrics}")
            
            # Calculate averages
            final_metrics = {}
            for metric_name, scores in all_metrics.items():
                if scores:
                    final_metrics[metric_name] = sum(scores) / len(scores)
                else:
                    final_metrics[metric_name] = 0.0
            
            # Add test case results to final metrics
            final_metrics['test_case_results'] = test_case_results
            
            logger.info("✅ RAGAS evaluation completed successfully")
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            raise
    
    def _calculate_final_metrics(self, evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """Calculate final metrics and determine overall status"""
        try:
            # Get target metrics from config
            target_metrics = self.config.get_target_metrics()
            
            # Calculate which targets are met
            met_targets = 0
            requirements = {}
            
            for metric_name, target_value in target_metrics.items():
                current_value = evaluation_results.get(metric_name, 0.0)
                status = "PASS" if current_value >= target_value else "FAIL"
                
                if status == "PASS":
                    met_targets += 1
                
                requirements[metric_name] = {
                    "current": current_value,
                    "target": target_value,
                    "status": status
                }
            
            # Determine overall status
            if met_targets == len(target_metrics):
                overall_status = "PASS"
            elif met_targets > 0:
                overall_status = "PARTIAL"
            else:
                overall_status = "FAIL"
            
            # Prepare final results
            final_results = {
                "test_cases": self.ragas_testset,
                "average_metrics": evaluation_results,
                "requirements": requirements,
                "total_cases": len(self.ragas_testset),
                "overall_status": overall_status,
                "met_targets": met_targets,
                "evaluation_strategy": "ragas_current_api"
            }
            
            logger.info(f"🎯 Overall status: {overall_status} ({met_targets}/{len(target_metrics)} targets met)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error calculating final metrics: {e}")
            raise
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        try:
            filename = 'data/eval/evaluation_results.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Evaluation results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving evaluation results: {e}")
            raise
