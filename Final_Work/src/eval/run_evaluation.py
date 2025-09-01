#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner
Orchestrates all evaluation components with proper logging and configuration
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger, log_system_start, log_system_stop, log_performance
from core.config_manager import ConfigManager
from eval.config import EvaluationConfigManager
from eval.ground_truth_manager import GroundTruthManager
from eval.metrics_calculator import MetricsCalculator
from eval.test_set_generator import TestSetGenerator
from eval.ragas_evaluator import RAGASEvaluator
from typing import Dict, Any


class EvaluationRunner:
    """Comprehensive evaluation runner for the entire system"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger('evaluation_runner')
        
        # Initialize configuration
        self.main_config = ConfigManager(config_path)
        self.eval_config = EvaluationConfigManager(self.main_config)
        
        # Initialize components
        self.ground_truth_manager = GroundTruthManager(self.eval_config)
        self.metrics_calculator = MetricsCalculator(self.eval_config)
        self.test_set_generator = TestSetGenerator(self.eval_config)
        self.ragas_evaluator = RAGASEvaluator(self.eval_config)
        
        self.logger.info("🚀 Evaluation Runner initialized")
    
    def run_complete_evaluation_pipeline(self) -> bool:
        """Run the complete evaluation pipeline"""
        try:
            start_time = time.time()
            log_system_start('evaluation_runner', operation='complete_evaluation_pipeline')
            
            self.logger.info("🚀 Starting complete evaluation pipeline")
            
            # Step 1: Validate configuration
            if not self._validate_configuration():
                return False
            
            # Step 2: Validate ground truth data
            if not self._validate_ground_truth():
                return False
            
            # Step 3: Generate test sets
            if not self._generate_test_sets():
                return False
            
            # Step 4: Run RAGAS evaluation
            if not self._run_ragas_evaluation():
                return False
            
            # Step 5: Generate reports
            if not self._generate_reports():
                return False
            
            duration = time.time() - start_time
            log_performance('evaluation_runner', 'complete_pipeline', duration)
            
            self.logger.info("✅ Complete evaluation pipeline finished successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation pipeline failed: {e}")
            log_system_stop('evaluation_runner', reason='error', error=str(e))
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate evaluation configuration"""
        try:
            self.logger.info("🔍 Validating evaluation configuration...")
            
            if not self.eval_config.validate_configuration():
                self.logger.error("❌ Configuration validation failed")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation error: {e}")
            return False
    
    def _validate_ground_truth(self) -> bool:
        """Validate ground truth data"""
        try:
            self.logger.info("🔍 Validating ground truth data...")
            
            # First load the ground truth data
            self.ground_truth_manager.load_ground_truth()
            
            # Then validate it
            validation_results = self.ground_truth_manager.validate_ground_truth()
            
            if validation_results['invalid_entries'] > 0:
                self.logger.warning(f"⚠️ Found {validation_results['invalid_entries']} invalid entries")
                
                # Export validation report
                self.ground_truth_manager.export_validation_report()
                self.logger.info("📊 Validation report exported to data/eval/validation_report.json")
            
            self.logger.info("✅ Ground truth validation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ground truth validation error: {e}")
            return False
    
    def _generate_test_sets(self) -> bool:
        """Load evaluation questions as test set"""
        try:
            self.logger.info("🔧 Loading evaluation questions as test set...")
            
            # Load evaluation questions which contain the test cases
            try:
                with open('data/eval/evaluation_questions.json', 'r', encoding='utf-8') as f:
                    evaluation_questions = json.load(f)
                self.logger.info("✅ Loaded evaluation questions with test cases")
                
                # Convert evaluation questions to test set format
                testset_to_use = []
                for q in evaluation_questions:
                    test_case = {
                        'question': q['question'],
                        'contexts': q.get('contexts', [q.get('answer', '')]),
                        'answer': q.get('answer', ''),
                        'ground_truth': q.get('ground_truth', '')
                    }
                    testset_to_use.append(test_case)
                
            except FileNotFoundError:
                self.logger.error("❌ evaluation_questions.json not found")
                return False
            
            # Validate test set
            validation_results = self.test_set_generator.validate_test_set(testset_to_use)
            
            if validation_results['invalid_cases'] > 0:
                self.logger.warning(f"⚠️ Found {validation_results['invalid_cases']} invalid test cases")
            
            # Get summary of test set
            summary = self.test_set_generator.get_test_set_summary(testset_to_use)
            self.logger.info(f"✅ Loaded {summary['total_cases']} test cases")
            
            # Store the test set for evaluation
            self.current_testset = testset_to_use
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Test set loading error: {e}")
            return False
    
    def _run_ragas_evaluation(self) -> bool:
        """Run RAGAS evaluation"""
        try:
            self.logger.info("📊 Running RAGAS evaluation...")
            
            # Run evaluation using the enhanced evaluator with current test set
            if hasattr(self, 'current_testset') and self.current_testset:
                results = self.ragas_evaluator.run_complete_evaluation(self.current_testset)
            else:
                results = self.ragas_evaluator.run_complete_evaluation()
            
            if not results:
                self.logger.error("❌ RAGAS evaluation failed to produce results")
                return False
            
            self.logger.info("✅ RAGAS evaluation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RAGAS evaluation error: {e}")
            return False
    
    def _generate_reports(self) -> bool:
        """Generate comprehensive evaluation reports"""
        try:
            self.logger.info("📋 Generating evaluation reports...")
            
            # Generate ground truth summary
            gt_summary = self.ground_truth_manager.get_ground_truth_summary()
            
            # Generate test set summary from current test set
            if hasattr(self, 'current_testset') and self.current_testset:
                test_set_summary = self.test_set_generator.get_test_set_summary(self.current_testset)
            else:
                # Fallback to original test set
                with open('data/eval/ragas_testset.json', 'r', encoding='utf-8') as f:
                    ragas_testset = json.load(f)
                test_set_summary = self.test_set_generator.get_test_set_summary(ragas_testset)
            
            # Create comprehensive report
            comprehensive_report = {
                'evaluation_summary': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'configuration': {
                        'targets': self.eval_config.get_target_metrics(),
                        'strategy': self.eval_config.ragas.strategy
                    },
                    'ground_truth': gt_summary,
                    'test_set': test_set_summary
                },
                'files_generated': [
                    'data/eval/validation_report.json',
                    'data/eval/ragas_testset.json',
                    'data/eval/evaluation_results.json',
                    'data/eval/evaluation_report.json'
                ]
            }
            
            # Save comprehensive report
            with open('data/eval/evaluation_report.json', 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ Comprehensive report generated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Report generation error: {e}")
            return False
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get current evaluation status"""
        try:
            return {
                'configuration_valid': self.eval_config.validate_configuration(),
                'ground_truth_loaded': len(self.ground_truth_manager.ground_truth) > 0,
                'test_sets_generated': hasattr(self, 'current_testset') and self.current_testset is not None,
                'evaluation_completed': os.path.exists('data/eval/evaluation_results.json'),
                'reports_generated': os.path.exists('data/eval/evaluation_report.json')
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting evaluation status: {e}")
            return {}


def main():
    """Main function to run the evaluation pipeline"""
    try:
        # Initialize evaluation runner
        runner = EvaluationRunner()
        
        # Run complete pipeline
        success = runner.run_complete_evaluation_pipeline()
        
        if success:            
            # Log success to file
            runner.logger.info("🎉 SUCCESS! Complete evaluation pipeline finished.")
            runner.logger.info("📊 Check the generated reports for detailed analysis.")
            
            # Display results table
            try:
                import sys
                import os
                # Add src/eval to path for import
                sys.path.append(os.path.join(os.path.dirname(__file__)))
                from show_results import show_results
                print("\n" + "="*60)
                
                # Log the results to file as well
                runner.logger.info("="*60)
                runner.logger.info("📊 EVALUATION RESULTS SUMMARY")
                runner.logger.info("="*60)
                
                # Capture the show_results output and log it
                import io
                from contextlib import redirect_stdout
                
                # Capture the output
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):
                    show_results()
                
                # Get the captured output and log it
                captured_output = output_buffer.getvalue()
                runner.logger.info(captured_output)
                
                # Also display in console
                print(captured_output)
                
            except ImportError as e:
                print("📊 Results saved to data/eval/evaluation_results.json")
                runner.logger.warning(f"Could not import show_results: {e}")
            except Exception as e:
                print("📊 Results saved to data/eval/evaluation_results.json")
                runner.logger.error(f"Error displaying results: {e}")
        else:
            print("🔍 Check the logs for detailed error information.")
            runner.logger.error("❌ FAILED! Evaluation pipeline did not complete successfully.")
        
        # Show evaluation status
        status = runner.get_evaluation_status()
        runner.logger.info("📊 EVALUATION STATUS:")
        
        for component, status_value in status.items():
            status_icon = "✅" if status_value else "❌"
            status_line = f"   {status_icon} {component.replace('_', ' ').title()}: {status_value}"
            print(status_line)
            runner.logger.info(status_line)
        
    except Exception as e:
        # Try to log the error if possible
        try:
            if 'runner' in locals():
                runner.logger.error(f"❌ CRITICAL ERROR: {e}")
        except:
            pass


if __name__ == "__main__":
    main()
