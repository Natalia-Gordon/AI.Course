#!/usr/bin/env python3
"""
Evaluation Configuration Manager
Provides type-safe, validated configuration access for evaluation components
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from core.config_manager import ConfigManager


@dataclass
class RAGASTargets:
    """RAGAS evaluation targets"""
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGASTargets':
        return cls(
            context_precision=data.get('context_precision', 0.75),
            context_recall=data.get('context_recall', 0.70),
            faithfulness=data.get('faithfulness', 0.85),
            answer_relevancy=data.get('answer_relevancy', 0.80)
        )


@dataclass
class TestSetConfig:
    """Test set configuration"""
    min_questions: int
    max_questions: int
    include_tables: bool
    include_summaries: bool
    include_needle: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestSetConfig':
        return cls(
            min_questions=data.get('min_questions', 5),
            max_questions=data.get('max_questions', 20),
            include_tables=data.get('include_tables', True),
            include_summaries=data.get('include_summaries', True),
            include_needle=data.get('include_needle', True)
        )


@dataclass
class AnswerGenerationConfig:
    """Answer generation configuration"""
    max_length: int
    include_sources: bool
    semantic_alignment: bool
    context_expansion: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnswerGenerationConfig':
        return cls(
            max_length=data.get('max_length', 500),
            include_sources=data.get('include_sources', True),
            semantic_alignment=data.get('semantic_alignment', True),
            context_expansion=data.get('context_expansion', True)
        )


@dataclass
class GroundTruthConfig:
    """Ground truth configuration"""
    source: str
    validate_metadata: bool
    require_vector_ids: bool
    max_contexts_per_question: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthConfig':
        return cls(
            source=data.get('source', 'pinecone_chunks'),
            validate_metadata=data.get('validate_metadata', True),
            require_vector_ids=data.get('require_vector_ids', True),
            max_contexts_per_question=data.get('max_contexts_per_question', 3)
        )


@dataclass
class OutputConfig:
    """Output configuration"""
    results_file: str
    detailed_results: bool
    include_metrics: bool
    include_performance: bool
    log_evaluation_steps: bool
    log_performance_metrics: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfig':
        return cls(
            results_file=data.get('results_file', 'data/eval/evaluation_results.json'),
            detailed_results=data.get('detailed_results', True),
            include_metrics=data.get('include_metrics', True),
            include_performance=data.get('include_performance', True),
            log_evaluation_steps=data.get('log_evaluation_steps', True),
            log_performance_metrics=data.get('log_performance_metrics', True)
        )


@dataclass
class RAGASConfig:
    """RAGAS configuration"""
    targets: RAGASTargets
    strategy: str
    test_set: TestSetConfig
    answer_generation: AnswerGenerationConfig
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGASConfig':
        return cls(
            targets=RAGASTargets.from_dict(data.get('targets', {})),
            strategy=data.get('strategy', 'langchain_integrated'),
            test_set=TestSetConfig.from_dict(data.get('test_set', {})),
            answer_generation=AnswerGenerationConfig.from_dict(data.get('answer_generation', {}))
        )


class EvaluationConfigManager:
    """Manages evaluation-specific configuration"""
    
    def __init__(self, main_config: ConfigManager):
        self.main_config = main_config
        self.eval_config = self._load_evaluation_config()
    
    def _load_evaluation_config(self) -> Dict[str, Any]:
        """Load evaluation configuration from main config"""
        try:
            eval_data = self.main_config.get('evaluation', {})
            
            return {
                'ragas': RAGASConfig.from_dict(eval_data.get('ragas', {})),
                'ground_truth': GroundTruthConfig.from_dict(eval_data.get('ground_truth', {})),
                'output': OutputConfig.from_dict(eval_data.get('output', {}))
            }
        except Exception as e:
            # Fallback to defaults if evaluation config is missing
            return {
                'ragas': RAGASConfig.from_dict({}),
                'ground_truth': GroundTruthConfig.from_dict({}),
                'output': OutputConfig.from_dict({})
            }
    
    @property
    def ragas(self) -> RAGASConfig:
        """Get RAGAS configuration"""
        return self.eval_config['ragas']
    
    @property
    def ground_truth(self) -> GroundTruthConfig:
        """Get ground truth configuration"""
        return self.eval_config['ground_truth']
    
    @property
    def output(self) -> OutputConfig:
        """Get output configuration"""
        return self.eval_config['output']
    
    def get_target_metrics(self) -> Dict[str, float]:
        """Get target metrics for evaluation"""
        targets = self.ragas.targets
        return {
            'context_precision': targets.context_precision,
            'context_recall': targets.context_recall,
            'faithfulness': targets.faithfulness,
            'answer_relevancy': targets.answer_relevancy
        }
    
    def validate_configuration(self) -> bool:
        """Validate evaluation configuration"""
        try:
            # Check required sections
            required_sections = ['ragas', 'ground_truth', 'output']
            for section in required_sections:
                if section not in self.eval_config:
                    return False
            
            # Validate target metrics
            targets = self.ragas.targets
            if not (0.0 <= targets.context_precision <= 1.0):
                return False
            if not (0.0 <= targets.context_recall <= 1.0):
                return False
            if not (0.0 <= targets.faithfulness <= 1.0):
                return False
            if not (0.0 <= targets.answer_relevancy <= 1.0):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_logging_config(self) -> Dict[str, str]:
        """Get logging configuration for evaluation components"""
        return {
            'evaluation_system': 'INFO',
            'ground_truth_manager': 'INFO',
            'metrics_calculator': 'INFO',
            'test_set_generator': 'INFO'
        }
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from main config manager"""
        return self.main_config.get_openai_api_key()
