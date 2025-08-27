import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from utils.schema import TableMetadata, SectionType

class TableProcessor:
    """Process and extract tables from documents."""
    
    def __init__(self, output_dir: str = "data/processed/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_tables_from_text(self, text: str, file_name: str, page_number: int) -> List[TableMetadata]:
        """Extract table structures from text and create metadata."""
        tables = []
        
        # Split text into sections
        sections = text.split('\n\n')
        
        for section_idx, section in enumerate(sections):
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            
            # Check if this looks like a table
            if self._is_table_section(lines):
                table_data = self._parse_table_lines(lines)
                if table_data and len(table_data) >= 2:
                    table_id = f"table_{file_name.replace('.', '_')}_{page_number}_{section_idx}"
                    table_metadata = self._create_table_metadata(
                        table_id, file_name, page_number, table_data
                    )
                    
                    # Save table files
                    csv_path, md_path = self._save_table_files(table_id, table_data)
                    table_metadata.csv_path = str(csv_path)
                    table_metadata.markdown_path = str(md_path)
                    
                    tables.append(table_metadata)
        
        return tables
    
    def _is_table_section(self, lines: List[str]) -> bool:
        """Determine if a section contains table data."""
        if not lines:
            return False
        
        # Check for pipe separators (common table format)
        pipe_count = sum(1 for line in lines if '|' in line)
        if pipe_count >= 2:  # At least header and one data row
            return True
        
        # Check for tab-separated or comma-separated data
        if len(lines) >= 2:
            first_line = lines[0]
            second_line = lines[1]
            
            # Check for consistent separators
            if '\t' in first_line and '\t' in second_line:
                return True
            if ',' in first_line and ',' in second_line:
                # Make sure it's not just text with commas
                if len(first_line.split(',')) > 2 and len(second_line.split(',')) > 2:
                    return True
        
        return False
    
    def _parse_table_lines(self, lines: List[str]) -> Optional[List[List[str]]]:
        """Parse table lines into structured data."""
        if not lines:
            return None
        
        # Try to detect separator
        separator = self._detect_separator(lines[0])
        if not separator:
            return None
        
        table_data = []
        for line in lines:
            if separator in line:
                cells = [cell.strip() for cell in line.split(separator) if cell.strip()]
                if cells:
                    table_data.append(cells)
        
        # Ensure all rows have the same number of columns
        if table_data:
            max_cols = max(len(row) for row in table_data)
            normalized_data = []
            for row in table_data:
                normalized_row = row + [''] * (max_cols - len(row))
                normalized_data.append(normalized_row)
            return normalized_data
        
        return None
    
    def _detect_separator(self, line: str) -> Optional[str]:
        """Detect the separator used in the table."""
        separators = ['|', '\t', ',']
        
        for sep in separators:
            if sep in line:
                # Count separators to ensure it's not just text with the character
                parts = line.split(sep)
                if len(parts) > 2:  # At least 3 columns
                    return sep
        
        return None
    
    def _create_table_metadata(self, table_id: str, file_name: str, page_number: int, 
                              table_data: List[List[str]]) -> TableMetadata:
        """Create metadata for a table."""
        headers = table_data[0] if table_data else []
        rows = len(table_data) - 1 if len(table_data) > 1 else 0
        columns = len(headers)
        
        # Try to generate a caption/description
        caption = self._generate_table_caption(headers, table_data)
        
        return TableMetadata(
            table_id=table_id,
            file_name=file_name,
            page_number=page_number,
            section_type=SectionType.TABLE,
            rows=rows,
            columns=columns,
            headers=headers,
            caption=caption,
            description=caption
        )
    
    def _generate_table_caption(self, headers: List[str], data: List[List[str]]) -> str:
        """Generate a descriptive caption for the table."""
        if not headers:
            return "Table with data"
        
        # Look for common financial terms in headers
        financial_terms = ['revenue', 'income', 'expense', 'profit', 'loss', 'amount', 'value', 'price', 'cost']
        hebrew_financial = ['הכנסה', 'הוצאה', 'רווח', 'הפסד', 'סכום', 'ערך', 'מחיר', 'עלות']
        
        found_terms = []
        for header in headers:
            header_lower = header.lower()
            for term in financial_terms + hebrew_financial:
                if term in header_lower:
                    found_terms.append(term)
                    break
        
        if found_terms:
            return f"Financial data table containing: {', '.join(found_terms[:3])}"
        
        # Generic description based on structure
        if len(headers) <= 3:
            return f"Data table with {len(headers)} columns"
        else:
            return f"Comprehensive data table with {len(headers)} columns and {len(data)-1} rows"
    
    def _save_table_files(self, table_id: str, table_data: List[List[str]]) -> Tuple[Path, Path]:
        """Save table data as CSV and Markdown files."""
        # Save as CSV
        csv_path = self.output_dir / f"{table_id}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table_data)
        
        # Save as Markdown
        md_path = self.output_dir / f"{table_id}.md"
        with open(md_path, 'w', encoding='utf-8') as mdfile:
            mdfile.write(f"# {table_id}\n\n")
            
            # Write table in markdown format
            if table_data:
                headers = table_data[0]
                mdfile.write("| " + " | ".join(headers) + " |\n")
                mdfile.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                
                for row in table_data[1:]:
                    mdfile.write("| " + " | ".join(row) + " |\n")
        
        return csv_path, md_path
    
    def process_document_tables(self, text: str, file_name: str, page_number: int) -> List[Dict[str, Any]]:
        """Process all tables in a document and return metadata."""
        tables = self.extract_tables_from_text(text, file_name, page_number)
        
        # Convert to dictionary format for storage
        table_dicts = []
        for table in tables:
            table_dict = table.dict()
            table_dicts.append(table_dict)
        
        return table_dicts
    
    def get_table_summary(self, table_metadata: TableMetadata) -> str:
        """Generate a summary of the table for chunk creation."""
        summary_parts = [
            f"Table: {table_metadata.table_id}",
            f"Dimensions: {table_metadata.rows} rows × {table_metadata.columns} columns",
            f"Page: {table_metadata.page_number}",
        ]
        
        if table_metadata.caption:
            summary_parts.append(f"Content: {table_metadata.caption}")
        
        if table_metadata.headers:
            summary_parts.append(f"Columns: {', '.join(table_metadata.headers[:5])}")
            if len(table_metadata.headers) > 5:
                summary_parts.append(f"... and {len(table_metadata.headers) - 5} more")
        
        return " | ".join(summary_parts)
