#!/usr/bin/env python3
"""
Table Enhancer with LlamaExtract Integration
- Processes existing table files
- Uses LlamaExtract for table analysis
- Creates proper Table chunks with section_type="Table"
- Adds table content to Pinecone
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.schema import DocumentChunk, SectionType, Language, Audience, Permission, Collection
from ingest.data_loader import DataLoader

logger = logging.getLogger(__name__)

class TableEnhancer:
    """Enhance table processing with LlamaExtract and proper Pinecone integration."""
    
    def __init__(self, tables_dir: str = "data/processed/tables"):
        self.tables_dir = Path(tables_dir)
        # Create a minimal config for DataLoader
        config = {
            'documents_dir': 'data/documents',
            'processed_dir': 'data/processed'
        }
        self.data_loader = DataLoader(config)
        
    def process_all_tables(self) -> List[DocumentChunk]:
        """Process all existing table files and create enhanced chunks."""
        table_chunks = []
        
        if not self.tables_dir.exists():
            logger.error(f"Tables directory not found: {self.tables_dir}")
            return []
        
        # Get all CSV files
        csv_files = list(self.tables_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} table files to process")
        
        for csv_file in csv_files:
            try:
                table_chunk = self._process_single_table(csv_file)
                if table_chunk:
                    table_chunks.append(table_chunk)
                    logger.info(f"Processed table: {csv_file.name}")
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(table_chunks)} tables")
        return table_chunks
    
    def _process_single_table(self, csv_file: Path) -> Optional[DocumentChunk]:
        """Process a single table file and create an enhanced chunk."""
        try:
            # Read CSV data
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Extract table metadata from filename
            table_id = csv_file.stem
            file_name = self._extract_filename_from_table_id(table_id)
            page_number = self._extract_page_from_table_id(table_id)
            
            # Analyze table content
            table_analysis = self._analyze_table_content(df, table_id)
            
            # Create enhanced table chunk
            table_chunk = self._create_table_chunk(
                table_id=table_id,
                file_name=file_name,
                page_number=page_number,
                table_analysis=table_analysis,
                csv_file=csv_file
            )
            
            return table_chunk
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            return None
    
    def _extract_filename_from_table_id(self, table_id: str) -> str:
        """Extract original filename from table ID."""
        # table_document_pdf_157_1 -> document.pdf
        parts = table_id.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}.pdf"
        return "unknown.pdf"
    
    def _extract_page_from_table_id(self, table_id: str) -> int:
        """Extract page number from table ID."""
        # table_document_pdf_157_1 -> 157
        parts = table_id.split('_')
        if len(parts) >= 5:  # table_document_pdf_157_1
            try:
                return int(parts[4])  # parts[4] is the page number
            except ValueError:
                pass
        return 1
    
    def _analyze_table_content(self, df: pd.DataFrame, table_id: str) -> Dict[str, Any]:
        """Analyze table content and extract key information."""
        analysis = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'headers': df.columns.tolist(),
            'table_type': self._classify_table_type(df),
            'financial_indicators': self._extract_financial_indicators(df),
            'has_numeric_data': self._has_numeric_data(df),
            'has_hebrew_content': self._has_hebrew_content(df),
            'sample_data': df.head(3).to_dict('records')
        }
        
        return analysis
    
    def _classify_table_type(self, df: pd.DataFrame) -> str:
        """Classify table type based on content."""
        headers = ' '.join(df.columns.astype(str)).lower()
        
        # Financial performance tables
        financial_terms = ['revenue', 'income', 'profit', 'loss', 'הכנסה', 'רווח', 'הפסד']
        if any(term in headers for term in financial_terms):
            return 'financial_performance'
        
        # Balance sheet tables
        balance_terms = ['assets', 'liabilities', 'equity', 'נכסים', 'התחייבויות', 'הון']
        if any(term in headers for term in balance_terms):
            return 'balance_sheet'
        
        # Management discussion tables
        mgmt_terms = ['management', 'discussion', 'analysis', 'ניהול', 'דיון', 'ניתוח']
        if any(term in headers for term in mgmt_terms):
            return 'management_discussion'
        
        return 'general_data'
    
    def _extract_financial_indicators(self, df: pd.DataFrame) -> List[str]:
        """Extract financial indicators from table."""
        indicators = []
        headers = ' '.join(df.columns.astype(str)).lower()
        
        financial_mappings = {
            'revenue': ['revenue', 'income', 'הכנסה', 'תקבול'],
            'profit': ['profit', 'earnings', 'רווח', 'הכנסה נטו'],
            'assets': ['assets', 'equity', 'נכסים', 'הון'],
            'liabilities': ['liabilities', 'debt', 'התחייבויות', 'חוב']
        }
        
        for category, terms in financial_mappings.items():
            if any(term in headers for term in terms):
                indicators.append(category)
        
        return list(set(indicators))
    
    def _has_numeric_data(self, df: pd.DataFrame) -> bool:
        """Check if table contains numeric data."""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                return True
            # Check if string columns contain numbers
            if df[col].dtype == 'object':
                if df[col].astype(str).str.contains(r'\d').any():
                    return True
        return False
    
    def _has_hebrew_content(self, df: pd.DataFrame) -> bool:
        """Check if table contains Hebrew content."""
        hebrew_chars = 'אבגדהוזחטסעפצקרשת'
        for col in df.columns:
            if any(char in str(col) for char in hebrew_chars):
                return True
            if df[col].dtype == 'object':
                for value in df[col].astype(str):
                    if any(char in value for char in hebrew_chars):
                        return True
        return False
    
    def _create_table_chunk(self, table_id: str, file_name: str, page_number: int,
                           table_analysis: Dict[str, Any], csv_file: Path) -> DocumentChunk:
        """Create a DocumentChunk specifically for table content."""
        
        # Generate table summary
        table_summary = self._generate_table_summary(table_analysis)
        
        # Create table text content
        table_text = self._create_table_text_content(table_analysis, csv_file)
        
        # Create chunk with proper Table section type
        chunk = DocumentChunk(
            id=table_id,
            file_name=file_name,
            client_id=None,  # Will be populated by LlamaExtract
            case_id=None,
            page_number=page_number,
            section_type=SectionType.TABLE,  # Explicitly set as TABLE
            table_id=table_id,
            figure_id=None,
            row_idx=None,
            col_idx=None,
            chunk_index=0,
            chunk_tokens=len(table_text.split()),
            chunk_summary=table_summary,
            text=table_text,
            keywords=table_analysis['financial_indicators'],
            critical_entities=[],

            amount_range=None,
            language=Language.HEBREW if table_analysis['has_hebrew_content'] else Language.ENGLISH,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            audience=Audience.INTERNAL,
            permissions=[Permission.USER],
            collection=Collection.KNOWLEDGE_BASE,
            topic=[table_analysis['table_type']],
            uri=str(csv_file),
            freshness=1.0,
            priority=0.8  # Higher priority for tables
        )
        
        return chunk
    
    def _generate_table_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the table."""
        summary_parts = []
        
        # Table type and structure
        summary_parts.append(f"Table type: {analysis['table_type'].replace('_', ' ').title()}")
        summary_parts.append(f"Structure: {analysis['row_count']} rows × {analysis['column_count']} columns")
        
        # Headers summary
        if analysis['headers']:
            headers = analysis['headers']
            header_summary = ", ".join(str(h) for h in headers[:5])
            if len(headers) > 5:
                header_summary += f" and {len(headers) - 5} more columns"
            summary_parts.append(f"Columns: {header_summary}")
        
        # Financial indicators
        if analysis['financial_indicators']:
            indicators = ", ".join(analysis['financial_indicators'])
            summary_parts.append(f"Financial indicators: {indicators}")
        
        # Content characteristics
        if analysis['has_numeric_data']:
            summary_parts.append("Contains numeric data")
        
        if analysis['has_hebrew_content']:
            summary_parts.append("Contains Hebrew content")
        
        return ". ".join(summary_parts) + "."
    
    def _create_table_text_content(self, analysis: Dict[str, Any], csv_file: Path) -> str:
        """Create comprehensive text content for table chunk."""
        content_parts = []
        
        # Add table summary (generate it here since it's not in analysis yet)
        table_summary = self._generate_table_summary(analysis)
        content_parts.append(f"Table Summary: {table_summary}")
        
        # Add headers
        if analysis['headers']:
            content_parts.append(f"Table Headers: {', '.join(str(h) for h in analysis['headers'])}")
        
        # Add financial indicators
        if analysis['financial_indicators']:
            indicators = ', '.join(analysis['financial_indicators'])
            content_parts.append(f"Financial Indicators: {indicators}")
        
        # Add sample data
        if analysis['sample_data']:
            content_parts.append("Sample Data:")
            for i, row in enumerate(analysis['sample_data'][:3]):  # First 3 rows
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                content_parts.append(f"  Row {i+1}: {row_text}")
        
        # Add file reference
        content_parts.append(f"Source: {csv_file.name}")
        
        return "\n".join(content_parts)
    
    def enhance_with_llama_extract(self, table_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Enhance table chunks using LlamaExtract if available."""
        if not hasattr(self.data_loader, '_llama_extract'):
            logger.info("LlamaExtract not available, skipping enhancement")
            return table_chunks
        
        enhanced_chunks = []
        
        for chunk in table_chunks:
            try:
                # Extract financial data from table content using existing method
                # Use the CSV file path directly for LlamaExtract
                csv_file_path = str(chunk.uri) if chunk.uri else chunk.file_name
                extracted_data = self.data_loader.extract_financial_data(csv_file_path)
                
                if extracted_data:
                    # Enhance chunk with extracted data using existing method
                    chunk_dict = {
                        'id': chunk.id,
                        'file_name': chunk.file_name,
                        'client_id': chunk.client_id,
                        'case_id': chunk.case_id,
                        'page_number': chunk.page_number,
                        'section_type': chunk.section_type.value if hasattr(chunk.section_type, 'value') else chunk.section_type,
                        'table_id': chunk.table_id,
                        'figure_id': chunk.figure_id,
                        'row_idx': chunk.row_idx,
                        'col_idx': chunk.col_idx,
                        'chunk_index': chunk.chunk_index,
                        'chunk_tokens': chunk.chunk_tokens,
                        'chunk_summary': chunk.chunk_summary,
                        'text': chunk.text,
                        'keywords': chunk.keywords,
                        'critical_entities': chunk.critical_entities,

                        'amount_range': chunk.amount_range,
                        'language': chunk.language.value if hasattr(chunk.language, 'value') else chunk.language,
                        'topic': chunk.topic,
                        'uri': chunk.uri,
                        'freshness': chunk.freshness,
                        'priority': chunk.priority
                    }
                    
                    # Use existing enhance method
                    enhanced_chunk_dict = self.data_loader.enhance_chunks_with_financial_data([chunk_dict], extracted_data)[0]
                    
                    # Update the original chunk with enhanced data
                    if enhanced_chunk_dict.get('extracted_financial_data'):
                        chunk.extracted_financial_data = enhanced_chunk_dict['extracted_financial_data']
                    if enhanced_chunk_dict.get('client_id'):
                        chunk.client_id = enhanced_chunk_dict['client_id']
                    
                    logger.info(f"Enhanced table chunk {chunk.id} with LlamaExtract data")
                
                enhanced_chunks.append(chunk)
                
            except Exception as e:
                logger.error(f"Error enhancing chunk {chunk.id}: {e}")
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
