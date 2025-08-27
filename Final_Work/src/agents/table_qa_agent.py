import re
from typing import List, Dict
import pandas as pd

def run_table_qa(query: str, contexts: List[Dict]) -> str:
    """Answer questions about tables in financial documents."""
    
    # Extract table-related contexts
    table_contexts = []
    for context in contexts:
        if context.get("has_tables") or "|" in context.get("text", ""):
            table_contexts.append(context)
    
    if not table_contexts:
        return "No tables found in the retrieved content."
    
    # Analyze the query to determine what we're looking for
    query_lower = query.lower()
    analysis_type = determine_analysis_type(query_lower)
    
    answers = []
    
    for context in table_contexts:
        text = context.get("text", "")
        if not text:
            continue
            
        # Try to parse as table
        tables = extract_tables_from_text(text)
        
        for table_idx, table_data in enumerate(tables):
            if not table_data or len(table_data) < 2:
                continue
                
            try:
                df = create_dataframe(table_data)
                if df is not None and not df.empty:
                    table_answer = analyze_table(df, query_lower, analysis_type, context, table_idx)
                    if table_answer:
                        answers.append(table_answer)
            except Exception as e:
                continue
    
    if not answers:
        return "Could not extract meaningful table data to answer your question."
    
    return "\n\n".join(answers)

def determine_analysis_type(query: str) -> str:
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

def extract_tables_from_text(text: str) -> List[List[List[str]]]:
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

def create_dataframe(table_data: List[List[str]]) -> pd.DataFrame:
    """Create a pandas DataFrame from table data."""
    if not table_data or len(table_data) < 2:
        return None
    
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

def analyze_table(df: pd.DataFrame, query: str, analysis_type: str, context: Dict, table_idx: int) -> str:
    """Analyze the table and provide relevant answers."""
    if df is None or df.empty:
        return None
    
    ref = get_table_reference(context, table_idx)
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

def get_table_reference(context: Dict, table_idx: int) -> str:
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
