#!/usr/bin/env python3
"""
Table Processor - CSV/MD Conversion with Captions
Implements the requirement for table conversion and semantic retrieval enhancement
"""

import os
import sys
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.schema import DocumentChunk, SectionType, Language, Audience, Permission, Collection
from ingest.data_loader import DataLoader

logger = logging.getLogger(__name__)

class TableProcessor:
    """Process tables and convert them to CSV/MD with captions for semantic retrieval."""
    
    def __init__(self, output_dir: str = "data/processed/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.csv_dir = self.output_dir / "csv"
        self.md_dir = self.output_dir / "markdown"
        self.csv_dir.mkdir(exist_ok=True)
        self.md_dir.mkdir(exist_ok=True)
        
        # Initialize DataLoader
        config = {
            'documents_dir': 'data/documents',
            'processed_dir': 'data/processed'
        }
        self.data_loader = DataLoader(config)
    
    def extract_tables_from_text(self, text: str, chunk_id: str, page_number: int) -> List[Dict[str, Any]]:
        """Extract table-like structures from text and convert to structured format."""
        tables = []
        
        # Split text into lines
        lines = text.split('\n')
        current_table = []
        table_start = -1
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line contains table-like content
            if self._is_table_line(line):
                if not current_table:
                    table_start = i
                current_table.append(line)
            elif current_table:
                # End of table detected
                if len(current_table) >= 2:  # At least header + 1 data row
                    table_data = self._process_table_lines(current_table)
                    if table_data:
                        table_info = self._create_table_info(
                            table_data, chunk_id, page_number, i - len(current_table)
                        )
                        tables.append(table_info)
                
                current_table = []
        
        # Handle table at end of text
        if current_table and len(current_table) >= 2:
            table_data = self._process_table_lines(current_table)
            if table_data:
                table_info = self._create_table_info(
                    table_data, chunk_id, page_number, len(lines) - len(current_table)
                )
                tables.append(table_info)
        
        return tables
    
    def _is_table_line(self, line: str) -> bool:
        """Check if a line contains table-like content."""
        # Look for patterns that suggest table structure
        patterns = [
            r'\d+[,.]?\d*',  # Numbers
            r'\d+%',         # Percentages
            r'[A-Za-z]+\s+\d+',  # Text followed by numbers
            r'\d+\s+[A-Za-z]+',  # Numbers followed by text
        ]
        
        # Check if line has multiple columns (separated by spaces or special chars)
        if len(line.split()) >= 3:
            return True
        
        # Check for numeric patterns
        for pattern in patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def _process_table_lines(self, lines: List[str]) -> Optional[List[List[str]]]:
        """Process table lines into structured data."""
        if not lines or len(lines) < 2:
            return None
        
        # Try to identify column separators
        separators = ['  ', '\t', '|', ';', ',']
        best_separator = None
        max_columns = 0
        
        for sep in separators:
            for line in lines:
                columns = line.split(sep)
                if len(columns) > max_columns:
                    max_columns = len(columns)
                    best_separator = sep
        
        if not best_separator:
            # Fallback: split by multiple spaces
            best_separator = '  '
        
        # Process table data
        table_data = []
        for line in lines:
            if best_separator == '  ':
                # Split by multiple spaces
                columns = re.split(r'\s{2,}', line.strip())
            else:
                columns = line.split(best_separator)
            
            # Clean columns
            columns = [col.strip() for col in columns if col.strip()]
            if columns:
                table_data.append(columns)
        
        return table_data if len(table_data) >= 2 else None
    
    def _create_table_info(self, table_data: List[List[str]], chunk_id: str, page_number: int, line_start: int) -> Dict[str, Any]:
        """Create table information structure."""
        # Generate table ID
        table_id = f"table_{chunk_id}_{line_start}"
        
        # Analyze table structure
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        # Extract headers (first row)
        headers = table_data[0] if table_data else []
        
        # Generate caption
        caption = self._generate_table_caption(table_data, headers, rows, cols)
        
        return {
            'table_id': table_id,
            'chunk_id': chunk_id,
            'page_number': page_number,
            'line_start': line_start,
            'rows': rows,
            'columns': cols,
            'headers': headers,
            'data': table_data,
            'caption': caption,
            'created_at': datetime.now().isoformat()
        }
    
    def _generate_table_caption(self, table_data: List[List[str]], headers: List[str], rows: int, cols: int) -> str:
        """Generate a descriptive caption for the table."""
        caption_parts = []
        
        # Basic structure
        caption_parts.append(f"Table with {rows} rows and {cols} columns")
        
        # Analyze headers for content type
        if headers:
            header_text = ", ".join(headers[:3])
            if len(headers) > 3:
                header_text += f" and {len(headers) - 3} more columns"
            caption_parts.append(f"Columns: {header_text}")
        
        # Look for financial indicators
        financial_terms = ['revenue', 'income', 'assets', 'liabilities', 'הכנסה', 'רווח', 'נכסים', 'התחייבויות']
        found_terms = []
        
        for row in table_data:
            for cell in row:
                for term in financial_terms:
                    if term.lower() in str(cell).lower():
                        found_terms.append(term)
        
        if found_terms:
            unique_terms = list(set(found_terms))[:3]
            caption_parts.append(f"Contains financial data: {', '.join(unique_terms)}")
        
        # Count numeric values
        numeric_count = 0
        for row in table_data:
            for cell in row:
                if re.search(r'\d+', str(cell)):
                    numeric_count += 1
        
        if numeric_count > 0:
            caption_parts.append(f"Contains {numeric_count} numeric values")
        
        return ". ".join(caption_parts) + "."
    
    def convert_table_to_csv(self, table_info: Dict[str, Any]) -> str:
        """Convert table data to CSV format."""
        csv_file = self.csv_dir / f"{table_info['table_id']}.csv"
        
        try:
            # Create DataFrame
            df = pd.DataFrame(table_info['data'][1:], columns=table_info['headers'])
            
            # Save to CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            logger.info(f"Saved CSV: {csv_file}")
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error saving CSV {csv_file}: {e}")
            return None
    
    def convert_table_to_markdown(self, table_info: Dict[str, Any]) -> str:
        """Convert table data to Markdown format."""
        md_file = self.md_dir / f"{table_info['table_id']}.md"
        
        try:
            with open(md_file, 'w', encoding='utf-8') as f:
                # Write caption
                f.write(f"# {table_info['caption']}\n\n")
                
                # Write table
                f.write("| " + " | ".join(table_info['headers']) + " |\n")
                f.write("|" + "|".join(["---"] * len(table_info['headers'])) + "|\n")
                
                # Write data rows
                for row in table_info['data'][1:]:
                    # Pad row if needed
                    padded_row = row + [''] * (len(table_info['headers']) - len(row))
                    f.write("| " + " | ".join(str(cell) for cell in padded_row) + " |\n")
                
                # Write metadata
                f.write(f"\n**Table ID:** {table_info['table_id']}\n")
                f.write(f"**Page:** {table_info['page_number']}\n")
                f.write(f"**Rows:** {table_info['rows']}\n")
                f.write(f"**Columns:** {table_info['columns']}\n")
                f.write(f"**Created:** {table_info['created_at']}\n")
            
            logger.info(f"Saved Markdown: {md_file}")
            return str(md_file)
            
        except Exception as e:
            logger.error(f"Error saving Markdown {md_file}: {e}")
            return None
    
    def process_chunk_for_tables(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single chunk to extract and convert tables."""
        text = chunk.get('text', '')
        if not text:
            return []
        
        chunk_id = chunk.get('id', 'unknown')
        page_number = chunk.get('page_number', 0)
        
        # Extract tables from text
        tables = self.extract_tables_from_text(text, chunk_id, page_number)
        
        processed_tables = []
        for table_info in tables:
            try:
                # Convert to CSV
                csv_file = self.convert_table_to_csv(table_info)
                if csv_file:
                    table_info['csv_file'] = csv_file
                
                # Convert to Markdown
                md_file = self.convert_table_to_markdown(table_info)
                if md_file:
                    table_info['markdown_file'] = md_file
                
                processed_tables.append(table_info)
                
            except Exception as e:
                logger.error(f"Error processing table {table_info['table_id']}: {e}")
                continue
        
        return processed_tables
    
    def process_all_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all chunks to extract and convert tables."""
        all_tables = []
        
        logger.info(f"Processing {len(chunks)} chunks for table extraction...")
        
        for i, chunk in enumerate(chunks):
            try:
                tables = self.process_chunk_for_tables(chunk)
                all_tables.extend(tables)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks, found {len(all_tables)} tables")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        logger.info(f"Total tables extracted: {len(all_tables)}")
        return all_tables
    
    def create_table_chunks_for_pinecone(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create table chunks suitable for Pinecone indexing."""
        table_chunks = []
        
        for table in tables:
            # Create chunk metadata
            chunk = {
                'id': table['table_id'],
                'file_name': f"table_{table['table_id']}",
                'client_id': None,  # Will be populated by LlamaExtract
                'case_id': None,
                'page_number': table['page_number'],
                'section_type': 'Table',
                'table_id': table['table_id'],
                'chunk_summary': table['caption'],
                'text': self._table_to_text(table),
                'keywords': self._extract_table_keywords(table),
                'csv_file': table.get('csv_file'),
                'markdown_file': table.get('markdown_file'),
                'table_metadata': {
                    'rows': table['rows'],
                    'columns': table['columns'],
                    'headers': table['headers'],
                    'line_start': table['line_start']
                }
            }
            
            table_chunks.append(chunk)
        
        return table_chunks
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to readable text format."""
        lines = []
        
        # Add caption
        lines.append(table['caption'])
        lines.append("")
        
        # Add headers
        if table['headers']:
            lines.append("Headers: " + " | ".join(table['headers']))
        
        # Add sample data (first 5 rows)
        data_rows = table['data'][1:6]  # Skip header, take first 5 rows
        for i, row in enumerate(data_rows, 1):
            lines.append(f"Row {i}: " + " | ".join(str(cell) for cell in row))
        
        if len(table['data']) > 6:
            lines.append(f"... and {len(table['data']) - 6} more rows")
        
        return "\n".join(lines)
    
    def _extract_table_keywords(self, table: Dict[str, Any]) -> List[str]:
        """Extract keywords from table content."""
        keywords = []
        
        # Add table-specific keywords
        keywords.extend(['table', 'data', 'structured', 'financial'])
        
        # Add keywords from headers
        if table['headers']:
            keywords.extend(table['headers'][:5])
        
        # Add keywords from caption
        caption_words = table['caption'].lower().split()
        financial_terms = ['revenue', 'income', 'assets', 'liabilities', 'הכנסה', 'רווח', 'נכסים']
        for term in financial_terms:
            if term in caption_words:
                keywords.append(term)
        
        # Add numeric indicators
        if table['rows'] > 10:
            keywords.append('large_dataset')
        if table['columns'] > 5:
            keywords.append('wide_table')
        
        return list(set(keywords))[:10]  # Remove duplicates, limit to 10
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processed tables."""
        csv_files = list(self.csv_dir.glob("*.csv"))
        md_files = list(self.md_dir.glob("*.md"))
        
        return {
            'total_csv_files': len(csv_files),
            'total_md_files': len(md_files),
            'csv_files': [f.name for f in csv_files],
            'md_files': [f.name for f in md_files],
            'output_directory': str(self.output_dir)
        }
