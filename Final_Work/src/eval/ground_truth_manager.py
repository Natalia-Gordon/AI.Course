#!/usr/bin/env python3
"""
Ground Truth Manager for Evaluation System
Handles loading, validation, and management of ground truth data
"""

import json
import os
from typing import Dict, List, Any, Optional
from utils.logger import get_logger, log_agent_action, log_performance
from eval.config import EvaluationConfigManager


class GroundTruthManager:
    """Manages ground truth data for evaluation"""
    
    def __init__(self, config: EvaluationConfigManager):
        self.config = config
        self.logger = get_logger('ground_truth_manager')
        self.ground_truth: Dict[str, Dict] = {}
        self.pinecone_chunks: Dict[str, Any] = {}
        
        self.logger.info("🚀 Ground Truth Manager initialized")
    
    def load_ground_truth(self) -> Dict[str, Dict]:
        """Load ground truth data from file"""
        try:
            log_agent_action('ground_truth_manager', 'Loading ground truth data')
            
            with open('data/eval/ground_truth.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to indexed format
            self.ground_truth = {item['qid']: item for item in data}
            
            self.logger.info(f"✅ Loaded {len(self.ground_truth)} ground truth entries")
            log_performance('ground_truth_manager', 'load_ground_truth', 0.0, 
                          entries_loaded=len(self.ground_truth))
            
            return self.ground_truth
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load ground truth: {e}")
            return {}
    
    def load_pinecone_chunks(self) -> Dict[str, Any]:
        """Load Pinecone chunk samples for reference"""
        try:
            log_agent_action('ground_truth_manager', 'Loading Pinecone chunk samples')
            
            if os.path.exists('data/eval/pinecone_chunk_samples.json'):
                with open('data/eval/pinecone_chunks.json', 'r', encoding='utf-8') as f:
                    self.pinecone_chunks = json.load(f)
                
                self.logger.info("✅ Loaded Pinecone chunk samples")
            else:
                self.logger.warning("⚠️ Pinecone chunk samples not found")
                self.pinecone_chunks = {}
            
            return self.pinecone_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Pinecone chunks: {e}")
            return {}
    
    def validate_ground_truth(self) -> Dict[str, Any]:
        """Validate ground truth data against configuration requirements"""
        try:
            log_agent_action('ground_truth_manager', 'Validating ground truth data')
            
            validation_results = {
                'total_entries': len(self.ground_truth),
                'valid_entries': 0,
                'invalid_entries': 0,
                'validation_errors': [],
                'metadata_validation': {},
                'vector_id_validation': {}
            }
            
            for qid, entry in self.ground_truth.items():
                entry_valid = True
                entry_errors = []
                
                # Check required fields (handle both old and new formats)
                required_fields = ['ground_truth']
                optional_fields = ['question', 'contexts', 'supports']
                
                # Check required fields
                for field in required_fields:
                    if field not in entry:
                        entry_errors.append(f"Missing required field: {field}")
                        entry_valid = False
                
                # Check that at least one of the optional fields exists
                has_optional_fields = any(field in entry for field in optional_fields)
                if not has_optional_fields:
                    entry_errors.append("Missing at least one of: question, contexts, or supports")
                    entry_valid = False
                
                # Validate metadata if required
                if self.config.ground_truth.validate_metadata:
                    metadata_validation = self._validate_metadata(entry)
                    if not metadata_validation['valid']:
                        entry_errors.extend(metadata_validation['errors'])
                        entry_valid = False
                    
                    validation_results['metadata_validation'][qid] = metadata_validation
                
                # Validate vector IDs if required
                if self.config.ground_truth.require_vector_ids:
                    vector_id_validation = self._validate_vector_ids(entry)
                    if not vector_id_validation['valid']:
                        entry_errors.extend(vector_id_validation['errors'])
                        entry_valid = False
                    
                    validation_results['vector_id_validation'][qid] = vector_id_validation
                
                # Update validation results
                if entry_valid:
                    validation_results['valid_entries'] += 1
                else:
                    validation_results['invalid_entries'] += 1
                    validation_results['validation_errors'].append({
                        'qid': qid,
                        'errors': entry_errors
                    })
            
            # Log validation results
            self.logger.info(f"✅ Ground truth validation completed:")
            self.logger.info(f"   Valid entries: {validation_results['valid_entries']}")
            self.logger.info(f"   Invalid entries: {validation_results['invalid_entries']}")
            
            if validation_results['validation_errors']:
                self.logger.warning(f"⚠️ Found {len(validation_results['validation_errors'])} validation errors")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Ground truth validation failed: {e}")
            return {
                'total_entries': 0,
                'valid_entries': 0,
                'invalid_entries': 0,
                'validation_errors': [{'error': str(e)}],
                'metadata_validation': {},
                'vector_id_validation': {}
            }
    
    def _validate_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata for a ground truth entry"""
        validation = {'valid': True, 'errors': []}
        
        # Check if metadata exists in supports (new format) or directly (old format)
        metadata = None
        if 'supports' in entry and entry['supports']:
            # New format: metadata is in supports
            metadata = entry['supports'][0].get('metadata', {})
        elif 'metadata' in entry:
            # Old format: metadata is directly in entry
            metadata = entry['metadata']
        else:
            validation['errors'].append("Missing metadata section (neither in supports nor directly)")
            validation['valid'] = False
            return validation
        
        # Check required metadata fields
        required_metadata = ['section_type', 'page_number']
        for field in required_metadata:
            if field not in metadata:
                validation['errors'].append(f"Missing metadata field: {field}")
                validation['valid'] = False
        
        # Validate section type
        if 'section_type' in metadata:
            valid_section_types = ['Summary', 'Table', 'Financial_Results', 'Operational_Review']
            if metadata['section_type'] not in valid_section_types:
                validation['errors'].append(f"Invalid section_type: {metadata['section_type']}")
                validation['valid'] = False
        
        return validation
    
    def _validate_vector_ids(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vector IDs for a ground truth entry"""
        validation = {'valid': True, 'errors': []}
        
        # Check if supports (new format) or contexts (old format) exist
        supports_or_contexts = None
        if 'supports' in entry and entry['supports']:
            supports_or_contexts = entry['supports']
        elif 'contexts' in entry:
            supports_or_contexts = entry['contexts']
        else:
            validation['errors'].append("Missing supports or contexts section")
            validation['valid'] = False
            return validation
        
        if not isinstance(supports_or_contexts, list):
            validation['errors'].append("Supports/contexts must be a list")
            validation['valid'] = False
            return validation
        
        # Check each support/context for required fields
        for i, item in enumerate(supports_or_contexts):
            if not isinstance(item, dict):
                validation['errors'].append(f"Item {i} must be a dictionary")
                validation['valid'] = False
                continue
            
            # Check for pinecone_id (new format) or vector_id (old format)
            if 'pinecone_id' not in item and 'vector_id' not in item:
                validation['errors'].append(f"Item {i} missing pinecone_id or vector_id")
                validation['valid'] = False
            
            # Check for text or text_preview
            if 'text' not in item and 'text_preview' not in item:
                validation['errors'].append(f"Item {i} missing text or text_preview")
                validation['valid'] = False
        
        return validation
    
    def get_ground_truth_for_question(self, qid: str) -> Optional[Dict[str, Any]]:
        """Get ground truth data for a specific question"""
        return self.ground_truth.get(qid)
    
    def get_all_questions(self) -> List[str]:
        """Get list of all question IDs"""
        return list(self.ground_truth.keys())
    
    def get_ground_truth_summary(self) -> Dict[str, Any]:
        """Get summary statistics of ground truth data"""
        if not self.ground_truth:
            return {'total_questions': 0, 'question_types': {}}
        
        question_types = {}
        for entry in self.ground_truth.values():
            qtype = entry.get('type', 'unknown')
            question_types[qtype] = question_types.get(qtype, 0) + 1
        
        return {
            'total_questions': len(self.ground_truth),
            'question_types': question_types,
            'has_metadata': any('metadata' in entry for entry in self.ground_truth.values()),
            'has_vector_ids': any('contexts' in entry and 
                                any('vector_id' in ctx for ctx in entry['contexts']) 
                                for entry in self.ground_truth.values())
        }
    
    def export_validation_report(self, output_file: str = 'data/eval/validation_report.json'):
        """Export validation report to file"""
        try:
            validation_results = self.validate_ground_truth()
            summary = self.get_ground_truth_summary()
            
            report = {
                'validation_results': validation_results,
                'summary': summary,
                'configuration': {
                    'validate_metadata': self.config.ground_truth.validate_metadata,
                    'require_vector_ids': self.config.ground_truth.require_vector_ids,
                    'max_contexts_per_question': self.config.ground_truth.max_contexts_per_question
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Validation report exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export validation report: {e}")
            return False
