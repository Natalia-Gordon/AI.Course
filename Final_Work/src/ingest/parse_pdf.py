from pathlib import Path
from typing import List, Dict
import pdfplumber
import re
from utils.schema import SectionType

def extract_text_blocks(pdf_path: str) -> List[Dict]:
    """Extract text blocks with better structure detection for financial documents."""
    blocks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text() or ""
            
            # Detect section types based on content patterns
            section_type = detect_section_type(text)
            
            # Extract tables if present
            tables = page.extract_tables()
            table_texts = []
            for j, table in enumerate(tables):
                if table and any(any(cell for cell in row if cell) for row in table):
                    table_text = format_table_as_text(table)
                    table_texts.append(table_text)
            
            # Combine text and tables
            if table_texts:
                text = text + "\n\n" + "\n\n".join(table_texts)
            
            if text.strip():
                blocks.append({
                    "page_number": i,
                    "section_type": section_type,
                    "text": text.strip(),
                    "has_tables": len(table_texts) > 0,
                    "table_count": len(table_texts)
                })
    
    return blocks

def detect_section_type(text: str) -> SectionType:
    """Detect the type of section based on content patterns."""
    text_lower = text.lower()
    
    # Financial document patterns
    if any(word in text_lower for word in ["financial", "revenue", "profit", "loss", "income"]):
        return SectionType.ANALYSIS
    elif any(word in text_lower for word in ["summary", "overview", "executive"]):
        return SectionType.SUMMARY
    elif any(word in text_lower for word in ["table", "chart", "figure", "graph"]):
        return SectionType.TABLE
    elif any(word in text_lower for word in ["conclusion", "outlook", "forecast"]):
        return SectionType.CONCLUSION
    elif any(word in text_lower for word in ["risk", "management", "governance"]):
        return SectionType.ANALYSIS
    else:
        return SectionType.ANALYSIS

def format_table_as_text(table: List[List]) -> str:
    """Format table data as structured text."""
    if not table or not any(table):
        return ""
    
    # Filter out empty rows and cells
    filtered_table = []
    for row in table:
        filtered_row = [str(cell).strip() if cell else "" for cell in row]
        if any(cell for cell in filtered_row):
            filtered_table.append(filtered_row)
    
    if not filtered_table:
        return ""
    
    # Format as pipe-separated text
    lines = []
    for row in filtered_table:
        line = " | ".join(cell for cell in row if cell)
        if line.strip():
            lines.append(line)
    
    return "\n".join(lines)
