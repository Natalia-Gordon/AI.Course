#!/usr/bin/env python3
"""
Table QA Agent for Financial Documents
Handles questions about tables and financial data
"""

import re
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas not available, table analysis will be limited")

from utils.agent_logger import log_table_qa_agent
from utils.logger import get_logger
from core.config_manager import ConfigManager

class TableQAAgent:
    """Table QA Agent for analyzing financial tables and answering questions"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger('table_qa_agent')
        self.config = ConfigManager(config_path) if config_path else None
        self.logger.info("🚀 Table QA Agent initialized")
    
    @log_table_qa_agent
    def run_table_qa(self, query: str, contexts: List[Dict]) -> str:
        """Answer questions about tables in financial documents."""
        
        try:
            self.logger.info(f"🔍 Processing table QA query: {query[:100]}...")
            
            # Extract table-related contexts - look for our indexed table chunks
            table_contexts = []
            for context in contexts:
                # Check if this is a table chunk from our indexing
                metadata = context.get('metadata', {})
                if (metadata.get('section_type') == 'Table' or 
                    context.get('section_type') == 'Table' or
                    context.get('has_tables') or 
                    "|" in context.get("text", "")):
                    table_contexts.append(context)
    
                # If no table chunks found in retrieved contexts, try to find them directly
            if not table_contexts:
                try:
                    # Import here to avoid circular imports
                    from index.pinecone_index import PineconeIndex
                    
                    # Get index name from config
                    from core.config_manager import ConfigManager
                    config = ConfigManager()
                    index_name = config.pinecone.index_name if hasattr(config, 'pinecone') else 'hybrid-rag'
                    
                    # Search directly for table chunks
                    pinecone_index = PineconeIndex(index_name=index_name)
                    table_results = pinecone_index.search("table", k=20, namespace='ayalon_q1_2025')
                    
                    # Filter to actual table chunks
                    for result in table_results:
                        metadata = result.get('metadata', {})
                        if metadata.get('section_type') == 'Table':
                            table_contexts.append(result)
                    
                    if table_contexts:
                        self.logger.info(f"🔍 Found {len(table_contexts)} table chunks directly from Pinecone")
                    else:
                        return "No tables found in the retrieved content or in the database."
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not search Pinecone directly: {e}")
                    return "No tables found in the retrieved content."
            
            # Analyze the query to determine what we're looking for
            query_lower = query.lower()
            analysis_type = self.determine_analysis_type(query_lower)
            
            answers = []
            
            for context in table_contexts:
                text = context.get("text", "")
                metadata = context.get('metadata', {})
                
                # Check if this is our indexed table chunk
                if metadata.get('section_type') == 'Table' or context.get('section_type') == 'Table':
                    # This is our indexed table chunk - use the metadata and summary
                    table_answer = self.analyze_indexed_table(context, query_lower, analysis_type)
                    if table_answer:
                        answers.append(table_answer)
                elif text:
                    # Try to parse as traditional table format
                    tables = self.extract_tables_from_text(text)
                    
                    for table_idx, table_data in enumerate(tables):
                        if not table_data or len(table_data) < 2:
                            continue
                            
                        try:
                            df = self.create_dataframe(table_data)
                            if df is not None and not df.empty:
                                table_answer = self.analyze_table(df, query_lower, analysis_type, context, table_idx)
                                if table_answer:
                                    answers.append(table_answer)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error analyzing table {table_idx}: {e}")
                            continue
            
            if not answers:
                return "Could not extract meaningful table data to answer your question."
            
            # Create a comprehensive summary for table queries
            if 'revenue' in query_lower or 'financial' in query_lower or 'metrics' in query_lower:
                return self.create_comprehensive_table_summary(answers, query_lower)
            
            return "\n\n".join(answers)
            
        except Exception as e:
            self.logger.error(f"❌ Error in table QA: {e}")
            return f"Error processing table query: {str(e)}"

    def determine_analysis_type(self, query: str) -> str:
        """Determine what type of analysis the query is asking for."""
        if any(word in query for word in ["average", "avg", "mean"]):
            return "average"
        elif any(word in query for word in ["sum", "total", "summarize"]):
            return "sum"
        elif any(word in query for word in ["maximum", "max", "highest"]):
            return "max"
        elif any(word in query for word in ["minimum", "min", "lowest"]):
            return "min"
        elif any(word in query for word in ["count", "number"]):
            return "count"
        elif any(word in query for word in ["trend", "change", "growth"]):
            return "trend"
        else:
            return "general"

    def extract_tables_from_text(self, text: str) -> List[List[List[str]]]:
        """Extract table structures from text."""
        tables = []
        
        # Split by double newlines to find table sections
        sections = text.split('\n\n')
        
        for section in sections:
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
                
            # Check if this looks like a table
            if any('|' in line for line in lines):
                table_data = []
                for line in lines:
                    if '|' in line:
                        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if cells:
                            table_data.append(cells)
                
                if table_data and len(table_data) >= 2:
                    tables.append(table_data)
        
        return tables

    def analyze_indexed_table(self, context: Dict, query: str, analysis_type: str) -> str:
        """Analyze indexed table chunks from our Pinecone indexing."""
        
        metadata = context.get('metadata', {})
        table_id = metadata.get('table_id', context.get('table_id', 'Unknown'))
        chunk_summary = metadata.get('chunk_summary', context.get('chunk_summary', ''))
        text = context.get('text', '')
        
        # Extract key information from the table
        table_info = f"Table ID: {table_id}\n"
        
        if chunk_summary:
            table_info += f"Summary: {chunk_summary}\n"
        
        # Look for specific data based on query
        query_lower = query.lower()
        
        if 'revenue' in query_lower:
            # Look for revenue-related numbers in text
            revenue_patterns = [
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:NIS|₪|million|billion|thousand)',
                r'revenue[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                r'income[:\s]*(\d{1,3}(?:\.\d+)?)'
            ]
            
            for pattern in revenue_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    table_info += f"Revenue/Income figures found: {', '.join(matches[:5])}\n"
                    break
        
        if 'financial' in query_lower or 'metrics' in query_lower:
            # Look for financial metrics
            financial_patterns = [
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:NIS|₪|million|billion|thousand)',
                r'(\d+(?:\.\d+)?)\s*%',
                r'ratio[:\s]*(\d+(?:\.\d+)?)'
            ]
            
            all_numbers = []
            for pattern in financial_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                all_numbers.extend(matches)
            
            if all_numbers:
                table_info += f"Financial metrics found: {', '.join(all_numbers[:10])}\n"
        
        # Add table structure information
        if 'rows' in metadata or 'columns' in metadata:
            rows = metadata.get('rows', 'Unknown')
            cols = metadata.get('columns', 'Unknown')
            table_info += f"Table structure: {rows} rows × {cols} columns\n"
        
        return table_info

    def create_comprehensive_table_summary(self, answers: List[str], query: str) -> str:
        """Create a comprehensive summary of table data for financial queries."""
        
        summary = "📊 COMPREHENSIVE TABLE ANALYSIS\n"
        summary += "=" * 50 + "\n\n"
        
        # Count tables found
        summary += f"Found {len(answers)} relevant tables\n\n"
        
        # Group by table type and extract key information
        revenue_tables = []
        financial_tables = []
        
        for answer in answers:
            if 'revenue' in answer.lower() or 'income' in answer.lower():
                revenue_tables.append(answer)
            else:
                financial_tables.append(answer)
        
        if revenue_tables:
            summary += "💰 REVENUE & INCOME TABLES:\n"
            summary += "-" * 30 + "\n"
            for i, table in enumerate(revenue_tables[:3], 1):
                summary += f"{i}. {table}\n"
            summary += "\n"
        
        if financial_tables:
            summary += "📈 FINANCIAL METRICS TABLES:\n"
            summary += "-" * 30 + "\n"
            for i, table in enumerate(financial_tables[:3], 1):
                summary += f"{i}. {table}\n"
            summary += "\n"
        
        # Add overall summary
        summary += "📋 SUMMARY:\n"
        summary += "-" * 20 + "\n"
        summary += f"• Total tables analyzed: {len(answers)}\n"
        summary += f"• Revenue-focused tables: {len(revenue_tables)}\n"
        summary += f"• Financial metrics tables: {len(financial_tables)}\n"
        
        if 'revenue' in query:
            summary += "\n💡 For detailed revenue figures, examine the revenue tables above."
        
        return summary

    def create_dataframe(self, table_data: List[List[str]]) -> Optional[pd.DataFrame]:
        """Create a pandas DataFrame from table data."""
        if not PANDAS_AVAILABLE:
            return None
            
        if not table_data or len(table_data) < 2:
            return None
        
        try:
            # First row as headers
            headers = table_data[0]
            rows = table_data[1:]
            
            # Ensure all rows have the same number of columns
            max_cols = len(headers)
            normalized_rows = []
            
            for row in rows:
                normalized_row = row[:max_cols]
                # Pad with empty strings if needed
                while len(normalized_row) < max_cols:
                    normalized_row.append("")
                normalized_rows.append(normalized_row)
            
            df = pd.DataFrame(normalized_rows, columns=headers)
            
            # Try to convert numeric columns
            for col in df.columns:
                try:
                    # Remove currency symbols and commas
                    cleaned_col = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                    df[col] = pd.to_numeric(cleaned_col, errors='ignore')
                except Exception:
                    pass
            
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating DataFrame: {e}")
            return None

    def analyze_table(self, df: pd.DataFrame, query: str, analysis_type: str, context: Dict, table_idx: int) -> str:
        """Analyze the table and provide relevant answers."""
        if df is None or df.empty:
            return None
        
        ref = self.get_table_reference(context, table_idx)
        analysis_parts = [f"Table Analysis ({ref}):"]
        
        # Perform analysis based on type
        if analysis_type == "average":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    avg = df[col].mean()
                    if pd.notna(avg):
                        analysis_parts.append(f"Average {col}: {avg:.2f}")
                except Exception:
                    continue
        
        elif analysis_type == "sum":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    total = df[col].sum()
                    if pd.notna(total):
                        analysis_parts.append(f"Sum {col}: {total:.2f}")
                except Exception:
                    continue
        
        elif analysis_type == "max":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    max_val = df[col].max()
                    if pd.notna(max_val):
                        analysis_parts.append(f"Maximum {col}: {max_val:.2f}")
                except Exception:
                    continue
        
        elif analysis_type == "min":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    min_val = df[col].min()
                    if pd.notna(min_val):
                        analysis_parts.append(f"Minimum {col}: {min_val:.2f}")
                except Exception:
                    continue
        
        elif analysis_type == "count":
            analysis_parts.append(f"Number of rows: {len(df)}")
            analysis_parts.append(f"Number of columns: {len(df.columns)}")
        
        elif analysis_type == "trend":
            # Look for time-based columns
            time_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'year', 'quarter', 'month'])]
            if time_cols:
                analysis_parts.append(f"Time-based columns found: {', '.join(time_cols)}")
        
        # General table info
        analysis_parts.append(f"Table dimensions: {len(df)} rows × {len(df.columns)} columns")
        analysis_parts.append(f"Columns: {', '.join(df.columns)}")
        
        return "\n".join(analysis_parts)

    def get_table_reference(self, context: Dict, table_idx: int) -> str:
        """Generate a reference for the table."""
        ref_parts = []
        
        if context.get("page_number"):
            ref_parts.append(f"p{context['page_number']}")
        if context.get("section_type"):
            ref_parts.append(context["section_type"])
        if context.get("file_name"):
            ref_parts.append(context["file_name"])
        
        ref_parts.append(f"Table {table_idx + 1}")
        
        return " | ".join(ref_parts)


# Legacy function for backward compatibility
def run_table_qa(query: str, contexts: List[Dict]) -> str:
    """Legacy function for backward compatibility."""
    agent = TableQAAgent()
    return agent.run_table_qa(query, contexts)