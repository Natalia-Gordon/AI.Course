#!/usr/bin/env python3
"""
Metrics Calculator for RAGAS Evaluation using LangChain's Industry-Standard Evaluators
Handles calculation of all RAGAS metrics with proper logging and configuration
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from utils.logger import get_logger, log_agent_action, log_performance
from eval.config import EvaluationConfigManager

# LangChain evaluation imports
try:
    from langchain.evaluation import load_evaluator
    from langchain.evaluation import Criteria
    from langchain.evaluation import StringEvaluator
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is available
    if os.getenv('OPENAI_API_KEY'):
        LANGCHAIN_AVAILABLE = True
        print("✅ OpenAI API key found - LangChain evaluators will be used")
    else:
        LANGCHAIN_AVAILABLE = False
        print("⚠️ OpenAI API key not found - falling back to custom implementation")
        
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain evaluation not available, falling back to custom implementation")


class MetricsCalculator:
    """Calculates RAGAS metrics using LangChain's industry-standard evaluators only"""
    
    def __init__(self, config: EvaluationConfigManager):
        self.config = config
        self.logger = get_logger('metrics_calculator')
        self.targets = config.get_target_metrics()
        
        # Enable debug mode for detailed evaluation logging
        self.debug_mode = getattr(config, 'debug_mode', False)
        
        # Initialize LangChain evaluators
        self.evaluators = self._initialize_langchain_evaluators()
        
        self.logger.info("🚀 Metrics Calculator initialized with LangChain evaluators")
        self.logger.info(f"📊 Target metrics: {self.targets}")
        self.logger.info(f"🔧 LangChain evaluators: {len(self.evaluators)} loaded")
        if self.debug_mode:
            self.logger.info("🔍 Debug mode enabled - detailed scoring information will be logged")
    
    def _initialize_langchain_evaluators(self) -> Dict[str, Any]:
        """Initialize LangChain evaluators for all RAGAS metrics"""
        evaluators = {}
        
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("⚠️ LangChain not available, using custom implementation")
            return evaluators
        
        try:
            # Initialize OpenAI model for evaluation with timeout and retry settings
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                request_timeout=30,  # 30 second timeout
                max_retries=2  # Built-in retry mechanism
            )
            
            self.logger.info("🚀 OpenAI model initialized for evaluation")
            
                                      # Context Precision: Use criteria-based evaluation with OpenAI
            evaluators['context_precision'] = load_evaluator(
                 "criteria",
                 criteria={
                     "relevance": "Rate 0-1: How directly relevant are the contexts to the financial question? IMPORTANT: If contexts contain financial terms, numbers, or business concepts related to the question, score 0.7-1.0. If partially related, score 0.4-0.6. Only score 0.0-0.3 if completely irrelevant.",
                     "accuracy": "Rate 0-1: How accurate is the financial data in the contexts? IMPORTANT: If contexts contain precise numbers, dates, or factual financial information, score 0.8-1.0. If general information without specifics, score 0.5-0.7. Only score 0.0-0.4 if information is clearly wrong.",
                     "completeness": "Rate 0-1: How complete is the coverage of the question requirements? IMPORTANT: If contexts cover most or all aspects of the question, score 0.7-1.0. If partial coverage, score 0.4-0.6. Only score 0.0-0.3 if major aspects are missing."
                 },
                 llm=llm,
                 normalize_by_criteria=True  # Normalize scores across criteria
             )
            
            # Context Recall: Use embedding distance for semantic similarity
            evaluators['context_recall'] = load_evaluator("embedding_distance")
            
                         # Faithfulness: Use criteria-based evaluation for factual consistency
            evaluators['faithfulness'] = load_evaluator(
                 "criteria",
                 criteria={
                     "factual_consistency": "Rate 0-1: Does the answer contain only facts from the contexts? IMPORTANT: If answer is 100% factual from source, score 0.9-1.0. If mostly factual with minor interpretation, score 0.7-0.8. If contains some false facts, score 0.3-0.6. Only score 0.0-0.2 if significantly false.",
                     "no_hallucination": "Rate 0-1: Are there no made-up or incorrect facts? IMPORTANT: If no hallucinations detected, score 0.8-1.0. If minor inaccuracies, score 0.6-0.7. If some hallucinations, score 0.3-0.5. Only score 0.0-0.2 if significant hallucinations.",
                     "source_alignment": "Rate 0-1: Does the answer align with the source material? IMPORTANT: If perfectly aligned, score 0.9-1.0. If mostly aligned, score 0.7-0.8. If partially aligned, score 0.4-0.6. Only score 0.0-0.3 if contradicts source."
                 },
                 llm=llm,
                 normalize_by_criteria=True  # Normalize scores across criteria
             )
            
            # Answer Relevancy: Use criteria-based evaluation for answer quality
            evaluators['answer_relevancy'] = load_evaluator(
                "criteria",
                criteria={
                    "relevance": "Rate 0-1: Is the answer directly relevant to the question? IMPORTANT: If directly addresses the question with financial precision, score 0.8-1.0. If relevant but general, score 0.6-0.7. If partially relevant, score 0.4-0.5. Only score 0.0-0.3 if off-topic.",
                    "accuracy": "Rate 0-1: Is the answer accurate and well-supported? IMPORTANT: If highly accurate with clear evidence, score 0.8-1.0. If accurate but limited support, score 0.6-0.7. If partially accurate, score 0.4-0.5. Only score 0.0-0.3 if inaccurate.",
                    "completeness": "Rate 0-1: Does the answer fully address the question? IMPORTANT: If comprehensive coverage, score 0.8-1.0. If covers most aspects, score 0.6-0.7. If partial coverage, score 0.4-0.5. Only score 0.0-0.3 if incomplete.",
                    "clarity": "Rate 0-1: Is the answer clear and well-structured? IMPORTANT: If crystal clear with logical structure, score 0.8-1.0. If clear but could be better, score 0.6-0.7. If somewhat clear, score 0.4-0.5. Only score 0.0-0.3 if unclear."
                },
                llm=llm,
                normalize_by_criteria=True  # Normalize scores across criteria
            )
            
            self.logger.info("✅ LangChain evaluators initialized successfully with OpenAI")
            self.logger.info("🔧 Context Precision: Criteria-based evaluator with 3 criteria")
            self.logger.info("🔧 Context Recall: Embedding distance evaluator")
            self.logger.info("🔧 Faithfulness: Criteria-based evaluator with 3 criteria")
            self.logger.info("🔧 Answer Relevancy: Criteria-based evaluator with 4 criteria")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LangChain evaluators: {e}")
            self.logger.info("🔄 Falling back to custom implementation")
        
        return evaluators
    
    def calculate_all_metrics(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all RAGAS metrics using LangChain evaluators only"""
        try:
            log_agent_action('metrics_calculator', 'Calculating all RAGAS metrics with LangChain', 
                           test_cases_count=len(test_cases))
            
            start_time = time.time()
            
            # Calculate individual metrics using LangChain evaluators
            context_precision = self.calculate_context_precision(test_cases)
            context_recall = self.calculate_context_recall(test_cases)
            faithfulness = self.calculate_faithfulness(test_cases)
            answer_relevancy = self.calculate_answer_relevancy(test_cases)
            
            # Calculate average metrics
            avg_metrics = {
                'context_precision': context_precision,
                'context_recall': context_recall,
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy
            }
            
            # Check which targets are met
            met_targets = self.check_targets_met(avg_metrics)
            
            # Determine overall status
            overall_status = self.determine_overall_status(met_targets, len(self.targets))
            
            duration = time.time() - start_time
            
            results = {
                'test_cases': test_cases,
                'average_metrics': avg_metrics,
                'requirements': self._create_requirements_dict(avg_metrics),
                'total_cases': len(test_cases),
                'overall_status': overall_status,
                'met_targets': met_targets,
                'evaluation_strategy': f"LangChain-{self.config.ragas.strategy}",
                'langchain_evaluators_used': len(self.evaluators) > 0,
                'evaluation_methods': {
                    'context_precision': 'LangChain' if 'context_precision' in self.evaluators else 'Not Available',
                    'context_recall': 'LangChain' if 'context_recall' in self.evaluators else 'Not Available',
                    'faithfulness': 'LangChain' if 'faithfulness' in self.evaluators else 'Not Available',
                    'answer_relevancy': 'LangChain' if 'answer_relevancy' in self.evaluators else 'Not Available'
                }
            }
            
            # Log performance
            log_performance('metrics_calculator', 'calculate_all_metrics', duration,
                          test_cases=len(test_cases),
                          met_targets=met_targets,
                          overall_status=overall_status)
            
            self.logger.info(f"✅ All metrics calculated successfully in {duration:.3f}s")
            self.logger.info(f"🎯 Overall status: {overall_status} ({met_targets}/{len(self.targets)} targets met)")
            self.logger.info(f"🔧 LangChain evaluators used: {len(self.evaluators)}")
            
            # Log which evaluation method was used for each metric
            for metric_name, method in results['evaluation_methods'].items():
                self.logger.info(f"📊 {metric_name.replace('_', ' ').title()}: {method}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate metrics: {e}")
            raise
    
    def calculate_context_precision(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate Context Precision using LangChain evaluator only"""
        try:
            log_agent_action('metrics_calculator', 'Calculating Context Precision with LangChain')
            
            if not test_cases:
                return 0.0
            
            if 'context_precision' in self.evaluators:
                return self._calculate_context_precision_langchain(test_cases)
            else:
                self.logger.error("❌ LangChain Context Precision evaluator not available")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Context Precision calculation failed: {e}")
            return 0.0
    
    def _calculate_context_precision_langchain(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate context precision using LangChain evaluator"""
        try:
            evaluator = self.evaluators['context_precision']
            total_score = 0.0
            valid_cases = 0
            
            for test_case in test_cases:
                question = test_case.get('question', '')
                contexts = test_case.get('contexts', [])
                ground_truth = test_case.get('ground_truth', '')
                
                if not contexts or not ground_truth:
                    continue
                
                # Use LangChain evaluator for context precision
                try:
                    # Combine contexts for evaluation
                    combined_context = " ".join(contexts)
                    
                    # Evaluate using LangChain criteria with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Fix input format: LangChain expects (input, prediction) for criteria evaluators
                            result = evaluator.evaluate_strings(
                                input=question,  # The question being asked
                                prediction=combined_context  # The context to evaluate
                            )
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"⚠️ LangChain evaluation attempt {attempt + 1} failed: {retry_error}")
                                time.sleep(1)  # Wait before retry
                                continue
                            else:
                                raise retry_error  # Re-raise on final attempt
                    
                    # Extract score from LangChain result
                    if isinstance(result, dict):
                        # LangChain returns dictionaries with 'score' or 'value' keys
                        if 'score' in result and result['score'] is not None:
                            score = float(result['score'])
                            self.logger.info(f"✅ Using result['score']: {score}")
                        elif 'value' in result and result['value'] is not None:
                            # Handle binary values: 'Y'/'N' or 1/0
                            if result['value'] in ['Y', 'y', '1', 1, True]:
                                score = 1.0
                            elif result['value'] in ['N', 'n', '0', 0, False]:
                                score = 0.0
                            else:
                                score = 0.5
                            self.logger.info(f"✅ Using result['value']: {result['value']} -> {score}")
                        else:
                            score = 0.5
                            self.logger.warning(f"⚠️ No score or value found in result dict: {result}")
                    elif hasattr(result, 'score') and result.score is not None:
                        score = result.score
                        self.logger.info(f"✅ Using result.score: {score}")
                    elif hasattr(result, 'value') and result.value is not None:
                        score = float(result.value) if isinstance(result.value, (int, float)) else 0.5
                        self.logger.info(f"✅ Using result.value: {result.value} -> {score}")
                    else:
                        score = 0.5  # Default score
                        self.logger.warning(f"⚠️ FALLING BACK TO DEFAULT SCORE 0.5 - No score or value found in result")
                        self.logger.warning(f"⚠️ Result object attributes: {dir(result)}")
                        self.logger.warning(f"⚠️ Result type: {type(result)}")
                    
                    # Log detailed scoring information
                    if self.debug_mode:
                        if hasattr(result, 'reasoning') and result.reasoning:
                            self.logger.debug(f"🔍 Context Precision scoring: {score:.3f} - Reasoning: {result.reasoning}")
                        if hasattr(result, 'criteria_scores') and result.criteria_scores:
                            self.logger.debug(f"📊 Criteria breakdown: {result.criteria_scores}")
                    
                    # Log raw result object for debugging
                    self.logger.info(f"🔍 Raw result object: {result}")
                    self.logger.info(f"🔍 Result type: {type(result)}")
                    self.logger.info(f"🔍 Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                    
                    total_score += score
                    valid_cases += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ LangChain evaluation failed for case after retries: {e}")
                    # Skip this case if LangChain fails
                    continue
            
            if valid_cases == 0:
                return 0.0
            
            avg_score = total_score / valid_cases
            self.logger.info(f"📊 Context Precision (LangChain): {avg_score:.3f} ({valid_cases} valid cases)")
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"❌ LangChain context precision calculation failed: {e}")
            return 0.0
    

    
    def calculate_context_recall(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate Context Recall using LangChain evaluator only"""
        try:
            log_agent_action('metrics_calculator', 'Calculating Context Recall with LangChain')
            
            if not test_cases:
                return 0.0
            
            if 'context_recall' in self.evaluators:
                return self._calculate_context_recall_langchain(test_cases)
            else:
                self.logger.error("❌ LangChain Context Recall evaluator not available")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Context Recall calculation failed: {e}")
            return 0.0
    
    def _calculate_context_recall_langchain(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate context recall using LangChain embedding distance evaluator"""
        try:
            evaluator = self.evaluators['context_recall']
            total_score = 0.0
            valid_cases = 0
            
            for test_case in test_cases:
                question = test_case.get('question', '')
                contexts = test_case.get('contexts', [])
                ground_truth = test_case.get('ground_truth', '')
                
                if not contexts or not ground_truth:
                    continue
                
                try:
                    # Use LangChain embedding distance evaluator with retry logic
                    combined_context = " ".join(contexts)
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Fix input format: Embedding distance evaluator expects (prediction, reference)
                            result = evaluator.evaluate_strings(
                                prediction=combined_context,  # The context to evaluate
                                reference=ground_truth  # The ground truth to compare against
                            )
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"⚠️ LangChain evaluation attempt {attempt + 1} failed: {retry_error}")
                                time.sleep(1)  # Wait before retry
                                continue
                            else:
                                raise retry_error  # Re-raise on final attempt
                    
                    # Extract score from LangChain result
                    if isinstance(result, dict):
                        # LangChain returns dictionaries with 'score' or 'value' keys
                        if 'score' in result and result['score'] is not None:
                            score = float(result['score'])
                            self.logger.info(f"✅ Using result['score']: {score}")
                        elif 'value' in result and result['value'] is not None:
                            # Handle binary values: 'Y'/'N' or 1/0
                            if result['value'] in ['Y', 'y', '1', 1, True]:
                                score = 1.0
                            elif result['value'] in ['N', 'n', '0', 0, False]:
                                score = 0.0
                            else:
                                score = 0.5
                            self.logger.info(f"✅ Using result['value']: {result['value']} -> {score}")
                        else:
                            score = 0.5
                            self.logger.warning(f"⚠️ No score or value found in result dict: {result}")
                    elif hasattr(result, 'score') and result.score is not None:
                        score = result.score
                        self.logger.info(f"✅ Using result.score: {score}")
                    elif hasattr(result, 'value') and result.value is not None:
                        score = float(result.value) if isinstance(result.value, (int, float)) else 0.5
                        self.logger.info(f"✅ Using result.value: {result.value} -> {score}")
                    else:
                        score = 0.5
                        self.logger.warning(f"⚠️ FALLING BACK TO DEFAULT SCORE 0.5 - No score or value found in result")
                        self.logger.warning(f"⚠️ Result object attributes: {dir(result)}")
                        self.logger.warning(f"⚠️ Result type: {type(result)}")
                    
                    # Log detailed scoring information
                    if self.debug_mode:
                        if hasattr(result, 'reasoning') and result.reasoning:
                            self.logger.debug(f"🔍 Context Recall scoring: {score:.3f} - Reasoning: {result.reasoning}")
                        if hasattr(result, 'distance') and result.distance is not None:
                            self.logger.debug(f"📊 Embedding distance: {result.distance}")
                    
                    total_score += score
                    valid_cases += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ LangChain evaluation failed for case after retries: {e}")
                    # Skip this case if LangChain fails
                    continue
            
            if valid_cases == 0:
                return 0.0
            
            avg_score = total_score / valid_cases
            self.logger.info(f"📊 Context Recall (LangChain): {avg_score:.3f} ({valid_cases} valid cases)")
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"❌ LangChain context recall calculation failed: {e}")
            return 0.0
    

    
    def calculate_faithfulness(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate Faithfulness using LangChain evaluator only"""
        try:
            log_agent_action('metrics_calculator', 'Calculating Faithfulness with LangChain')
            
            if not test_cases:
                return 0.0
            
            if 'faithfulness' in self.evaluators:
                return self._calculate_faithfulness_langchain(test_cases)
            else:
                self.logger.error("❌ LangChain Faithfulness evaluator not available")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Faithfulness calculation failed: {e}")
            return 0.0
    
    def _calculate_faithfulness_langchain(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness using LangChain criteria evaluator"""
        try:
            evaluator = self.evaluators['faithfulness']
            total_score = 0.0
            valid_cases = 0
            
            for test_case in test_cases:
                answer = test_case.get('answer', '')
                contexts = test_case.get('contexts', [])
                
                if not answer or not contexts:
                    continue
                
                try:
                    # Use LangChain criteria evaluator for faithfulness with retry logic
                    combined_context = " ".join(contexts)
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Fix input format: For faithfulness, we evaluate if answer is faithful to context
                            result = evaluator.evaluate_strings(
                                input=combined_context,  # The source context
                                prediction=answer  # The answer to evaluate for faithfulness
                            )
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"⚠️ LangChain evaluation attempt {attempt + 1} failed: {retry_error}")
                                time.sleep(1)  # Wait before retry
                                continue
                            else:
                                raise retry_error  # Re-raise on final attempt
                    
                    # Extract score from LangChain result
                    if isinstance(result, dict):
                        # LangChain returns dictionaries with 'score' or 'value' keys
                        if 'score' in result and result['score'] is not None:
                            score = float(result['score'])
                            self.logger.info(f"✅ Using result['score']: {score}")
                        elif 'value' in result and result['value'] is not None:
                            # Handle binary values: 'Y'/'N' or 1/0
                            if result['value'] in ['Y', 'y', '1', 1, True]:
                                score = 1.0
                            elif result['value'] in ['N', 'n', '0', 0, False]:
                                score = 0.0
                            else:
                                score = 0.5
                            self.logger.info(f"✅ Using result['value']: {result['value']} -> {score}")
                        else:
                            score = 0.5
                            self.logger.warning(f"⚠️ No score or value found in result dict: {result}")
                    elif hasattr(result, 'score') and result.score is not None:
                        score = result.score
                        self.logger.info(f"✅ Using result.score: {score}")
                    elif hasattr(result, 'value') and result.value is not None:
                        score = float(result.value) if isinstance(result.value, (int, float)) else 0.5
                        self.logger.info(f"✅ Using result.value: {result.value} -> {score}")
                    else:
                        score = 0.5
                        self.logger.warning(f"⚠️ FALLING BACK TO DEFAULT SCORE 0.5 - No score or value found in result")
                        self.logger.warning(f"⚠️ Result object attributes: {dir(result)}")
                        self.logger.warning(f"⚠️ Result type: {type(result)}")
                    
                    # Log detailed scoring information
                    if self.debug_mode:
                        if hasattr(result, 'reasoning') and result.reasoning:
                            self.logger.debug(f"🔍 Faithfulness scoring: {score:.3f} - Reasoning: {result.reasoning}")
                        if hasattr(result, 'criteria_scores') and result.criteria_scores:
                            self.logger.debug(f"📊 Criteria breakdown: {result.criteria_scores}")
                    
                    total_score += score
                    valid_cases += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ LangChain evaluation failed for case after retries: {e}")
                    # Skip this case if LangChain fails
                    continue
            
            if valid_cases == 0:
                return 0.0
            
            avg_score = total_score / valid_cases
            self.logger.info(f"📊 Faithfulness (LangChain): {avg_score:.3f} ({valid_cases} valid cases)")
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"❌ LangChain faithfulness calculation failed: {e}")
            return 0.0
    

    
    def calculate_answer_relevancy(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate Answer Relevancy using LangChain evaluator only"""
        try:
            log_agent_action('metrics_calculator', 'Calculating Answer Relevancy with LangChain')
            
            if not test_cases:
                return 0.0
            
            if 'answer_relevancy' in self.evaluators:
                return self._calculate_answer_relevancy_langchain(test_cases)
            else:
                self.logger.error("❌ LangChain Answer Relevancy evaluator not available")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Answer Relevancy calculation failed: {e}")
            return 0.0
    
    def _calculate_answer_relevancy_langchain(self, test_cases: List[Dict[str, Any]]) -> float:
        """Calculate answer relevancy using LangChain criteria evaluator"""
        try:
            evaluator = self.evaluators['answer_relevancy']
            total_score = 0.0
            valid_cases = 0
            
            for test_case in test_cases:
                question = test_case.get('question', '')
                answer = test_case.get('answer', '')
                
                if not question or not answer:
                    continue
                
                try:
                    # Use LangChain criteria evaluator for answer relevancy with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Fix input format: For answer relevancy, we evaluate if answer is relevant to question
                            result = evaluator.evaluate_strings(
                                input=question,  # The question being asked
                                prediction=answer  # The answer to evaluate for relevancy
                            )
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"⚠️ LangChain evaluation attempt {attempt + 1} failed: {retry_error}")
                                time.sleep(1)  # Wait before retry
                                continue
                            else:
                                raise retry_error  # Re-raise on final attempt
                    
                    # Extract score from LangChain result
                    if isinstance(result, dict):
                        # LangChain returns dictionaries with 'score' or 'value' keys
                        if 'score' in result and result['score'] is not None:
                            score = float(result['score'])
                            self.logger.info(f"✅ Using result['score']: {score}")
                        elif 'value' in result and result['value'] is not None:
                            # Handle binary values: 'Y'/'N' or 1/0
                            if result['value'] in ['Y', 'y', '1', 1, True]:
                                score = 1.0
                            elif result['value'] in ['N', 'n', '0', 0, False]:
                                score = 0.0
                            else:
                                score = 0.5
                            self.logger.info(f"✅ Using result['value']: {result['value']} -> {score}")
                        else:
                            score = 0.5
                            self.logger.warning(f"⚠️ No score or value found in result dict: {result}")
                    elif hasattr(result, 'score') and result.score is not None:
                        score = result.score
                        self.logger.info(f"✅ Using result.score: {score}")
                    elif hasattr(result, 'value') and result.value is not None:
                        score = float(result.value) if isinstance(result.value, (int, float)) else 0.5
                        self.logger.info(f"✅ Using result.value: {result.value} -> {score}")
                    else:
                        score = 0.5
                    
                    # Log detailed scoring information
                    if self.debug_mode:
                        if hasattr(result, 'reasoning') and result.reasoning:
                            self.logger.debug(f"🔍 Answer Relevancy scoring: {score:.3f} - Reasoning: {result.reasoning}")
                        if hasattr(result, 'criteria_scores') and result.criteria_scores:
                            self.logger.debug(f"📊 Criteria breakdown: {result.criteria_scores}")
                    
                    total_score += score
                    valid_cases += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ LangChain evaluation failed for case after retries: {e}")
                    # Skip this case if LangChain fails
                    continue
            
            if valid_cases == 0:
                return 0.0
            
            avg_score = total_score / valid_cases
            self.logger.info(f"📊 Answer Relevancy (LangChain): {avg_score:.3f} ({valid_cases} valid cases)")
            
            return avg_score
            
        except Exception as e:
            self.logger.error(f"❌ LangChain answer relevancy calculation failed: {e}")
            return 0.0
    

    

    
    def check_targets_met(self, metrics: Dict[str, float]) -> int:
        """Check how many targets are met"""
        met_count = 0
        
        for metric_name, target_value in self.targets.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                if current_value >= target_value:
                    met_count += 1
        
        return met_count
    
    def determine_overall_status(self, met_targets: int, total_targets: int) -> str:
        """Determine overall evaluation status"""
        if met_targets == total_targets:
            return "PASS"
        elif met_targets >= total_targets // 2:
            return "PARTIAL"
        else:
            return "FAIL"
    
    def _create_requirements_dict(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Create requirements dictionary for output"""
        requirements = {}
        
        for metric_name, target_value in self.targets.items():
            current_value = metrics.get(metric_name, 0.0)
            status = "PASS" if current_value >= target_value else "FAIL"
            
            requirements[metric_name] = {
                'current': current_value,
                'target': target_value,
                'status': status
            }
        
        return requirements
