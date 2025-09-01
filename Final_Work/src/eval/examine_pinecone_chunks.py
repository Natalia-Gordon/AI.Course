#!/usr/bin/env python3
"""
Pinecone Chunks Examiner
Essential tool for examining Pinecone chunks and creating ground truth data
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger, log_agent_action, log_performance
from core.config_manager import ConfigManager
from index.pinecone_index import PineconeIndex


class PineconeChunksExaminer:
    """Examine Pinecone chunks for ground truth creation and analysis"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger('pinecone_examiner')
        self.config = ConfigManager(config_path)
        
        # Initialize PineconeIndex with proper configuration
        self.pinecone_index = PineconeIndex(
            api_key=os.environ.get("PINECONE_API_KEY"),
            region=os.environ.get("PINECONE_REGION"),
            cloud=os.environ.get("PINECONE_CLOUD", "aws"),
            index_name=self.config.pinecone.index_name,
            dimension=1536,
            metric=self.config.pinecone.metric
        )
        
        self.logger.info("🚀 Pinecone Chunks Examiner initialized")
    
    def examine_namespace_chunks(self, namespace: str = None) -> List[Dict[str, Any]]:
        """Examine chunks in a specific namespace"""
        try:
            start_time = time.time()
            log_agent_action('pinecone_examiner', 'Examining namespace chunks', namespace=namespace)
            
            if not namespace:
                namespace = "ayalon_q1_2025"  # Use the actual namespace from Pinecone
            
            self.logger.info(f"🔍 Examining chunks in namespace: {namespace}")
            
            # Get chunks from Pinecone
            chunks = self.pinecone_index.get_chunks_from_namespace(namespace)
            
            if not chunks:
                self.logger.warning(f"⚠️ No chunks found in namespace: {namespace}")
                return []
            
            self.logger.info(f"✅ Found {len(chunks)} chunks in namespace: {namespace}")
            
            # Analyze chunks
            analyzed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'id': chunk.get('id', f'chunk_{i}'),
                    'text': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {}),
                    'score': chunk.get('score', 0.0),
                    'section_type': chunk.get('metadata', {}).get('section_type', 'Unknown'),
                    'table_id': chunk.get('metadata', {}).get('table_id', None),
                    'row': chunk.get('metadata', {}).get('row', None),
                    'col': chunk.get('metadata', {}).get('col', None)
                }
                analyzed_chunks.append(chunk_info)
            
            duration = time.time() - start_time
            log_performance('pinecone_examiner', 'examine_namespace_chunks', duration, chunks_count=len(analyzed_chunks))
            
            self.logger.info(f"✅ Examined {len(analyzed_chunks)} chunks in {duration:.2f}s")
            return analyzed_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error examining namespace chunks: {e}")
            return []
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        try:
            if not chunks:
                return {}
            
            # Count by section type
            section_type_counts = {}
            table_chunks = []
            summary_chunks = []
            
            for chunk in chunks:
                section_type = chunk.get('section_type', 'Unknown')
                section_type_counts[section_type] = section_type_counts.get(section_type, 0) + 1
                
                if section_type == 'Table':
                    table_chunks.append(chunk)
                elif section_type == 'Summary':
                    summary_chunks.append(chunk)
            
            # Calculate text statistics
            text_lengths = [len(chunk.get('text', '')) for chunk in chunks]
            avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            statistics = {
                'total_chunks': len(chunks),
                'section_type_distribution': section_type_counts,
                'table_chunks_count': len(table_chunks),
                'summary_chunks_count': len(summary_chunks),
                'text_statistics': {
                    'average_length': avg_text_length,
                    'min_length': min(text_lengths) if text_lengths else 0,
                    'max_length': max(text_lengths) if text_lengths else 0
                }
            }
            
            self.logger.info(f"📊 Chunk statistics: {statistics}")
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating chunk statistics: {e}")
            return {}
    
    def export_chunk_samples(self, chunks: List[Dict[str, Any]], output_file: str = None) -> bool:
        """Export chunk samples for analysis"""
        try:
            if not output_file:
                output_file = f'data/eval/pinecone_chunk_samples_{int(time.time())}.json'
            
            # Prepare export data
            export_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_chunks': len(chunks),
                'chunks': chunks[:50],  # Export first 50 chunks as samples
                'statistics': self.get_chunk_statistics(chunks)
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Export to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Exported {len(chunks[:50])} chunk samples to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting chunk samples: {e}")
            return False
    
    def create_ground_truth_candidates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create ground truth candidates from chunks"""
        try:
            self.logger.info("🔧 Creating ground truth candidates from chunks")
            
            candidates = []
            for chunk in chunks:
                if chunk.get('section_type') in ['Summary', 'Table']:
                    candidate = {
                        'chunk_id': chunk.get('id'),
                        'text': chunk.get('text', ''),
                        'section_type': chunk.get('section_type'),
                        'metadata': chunk.get('metadata', {}),
                        'ground_truth': chunk.get('text', ''),  # Use chunk text as ground truth
                        'source': 'pinecone_chunk'
                    }
                    candidates.append(candidate)
            
            self.logger.info(f"✅ Created {len(candidates)} ground truth candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ground truth candidates: {e}")
            return []


def main():
    """Main function to examine Pinecone chunks"""
    try:
        # Initialize examiner
        examiner = PineconeChunksExaminer()
        
        # Examine chunks in default namespace
        chunks = examiner.examine_namespace_chunks()
        
        if chunks:
            # Get statistics
            stats = examiner.get_chunk_statistics(chunks)
            print(f"📊 Found {stats.get('total_chunks', 0)} chunks")
            print(f"📋 Section types: {stats.get('section_type_distribution', {})}")
            
            # Export samples
            examiner.export_chunk_samples(chunks)
            
            # Create ground truth candidates
            candidates = examiner.create_ground_truth_candidates(chunks)
            print(f"🎯 Created {len(candidates)} ground truth candidates")
            
        else:
            print("⚠️ No chunks found to examine")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
