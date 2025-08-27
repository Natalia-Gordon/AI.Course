from typing import List, Dict
import re

def clean_hebrew_text(text: str) -> str:
    """Clean and format Hebrew text for better readability."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common Hebrew text issues
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' :', ':')
    text = text.replace(' ;', ';')
    
    # Ensure proper spacing around Hebrew punctuation
    text = re.sub(r'([א-ת])([.,:;])', r'\1 \2', text)
    
    return text.strip()

def format_hebrew_summary(text: str, max_length: int = 300) -> str:
    """Format Hebrew text for summary display."""
    if not text:
        return ""
    
    # Clean the text
    cleaned = clean_hebrew_text(text)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        # Try to break at sentence boundary
        sentences = cleaned.split('.')
        if len(sentences) > 1:
            # Take first complete sentence
            return sentences[0].strip() + '.'
        else:
            # Break at word boundary
            words = cleaned.split()
            if len(words) > 10:
                return ' '.join(words[:10]) + '...'
    
    return cleaned

def run_summary(query: str, contexts: List[Dict]) -> str:
    """Generate a comprehensive summary based on retrieved contexts with LlamaExtract enhancement."""
    
    # Group contexts by section type
    by_section = {}
    for c in contexts:
        section = c.get("section_type", "Content")
        if section not in by_section:
            by_section[section] = []
        by_section[section].append(c)
    
    # Generate summary
    summary_parts = []
    
    # Enhanced executive summary using LlamaExtract data
    if "Summary" in by_section:
        summary_parts.append("EXECUTIVE SUMMARY:")
        for c in by_section["Summary"][:2]:  # Top 2 summary sections
            summ = c.get("chunk_summary") or c.get("text", "")
            formatted_summ = format_hebrew_summary(summ, 300)
            ref = get_reference(c)
            summary_parts.append(f"• {formatted_summ} ({ref})")
    
    # Enhanced financial highlights using LlamaExtract data
    financial_data = extract_llama_extract_financial_data(contexts)
    if financial_data:
        summary_parts.append("\nFINANCIAL HIGHLIGHTS (AI-Enhanced):")
        
        # Company and period info
        if financial_data.get('company_name'):
            summary_parts.append(f"• Company: {financial_data['company_name']}")
        if financial_data.get('report_period'):
            summary_parts.append(f"• Period: {financial_data['report_period']}")
        if financial_data.get('report_date'):
            summary_parts.append(f"• Date: {financial_data['report_date']}")
        
        # Financial metrics
        if financial_data.get('revenue'):
            summary_parts.append(f"• Revenue: {financial_data['revenue']}")
        if financial_data.get('net_income'):
            summary_parts.append(f"• Net Income: {financial_data['net_income']}")
        if financial_data.get('assets'):
            summary_parts.append(f"• Assets: {financial_data['assets']}")
        if financial_data.get('liabilities'):
            summary_parts.append(f"• Liabilities: {financial_data['liabilities']}")
        
        # KPIs
        if financial_data.get('kpis'):
            summary_parts.append(f"• Key KPIs: {', '.join(financial_data['kpis'][:5])}")
        
        # Executive insights
        if financial_data.get('executive_summary'):
            summary_parts.append(f"• AI Summary: {financial_data['executive_summary'][:200]}...")
        
        # Risk factors
        if financial_data.get('risk_factors'):
            summary_parts.append(f"• Risk Factors: {', '.join(financial_data['risk_factors'][:3])}")
        
        # Outlook
        if financial_data.get('outlook'):
            summary_parts.append(f"• Outlook: {financial_data['outlook'][:150]}...")
    
    # Key metrics and numbers (enhanced with LlamaExtract data)
    financial_metrics = extract_financial_highlights(contexts)
    if financial_metrics:
        summary_parts.append("\nKEY METRICS:")
        for metric in financial_metrics[:5]:
            summary_parts.append(f"• {metric}")
    
    # Other sections
    other_sections = [s for s in by_section.keys() if s not in ["Summary", "Financial"]]
    for section in other_sections[:2]:  # Top 2 other sections
        summary_parts.append(f"\n{section.upper()}:")
        for c in by_section[section][:2]:
            summ = c.get("chunk_summary") or c.get("text", "")
            formatted_summ = format_hebrew_summary(summ, 200)
            ref = get_reference(c)
            summary_parts.append(f"• {formatted_summ} ({ref})")
    
    return "\n".join(summary_parts)

def extract_llama_extract_financial_data(contexts: List[Dict]) -> Dict:
    """Extract LlamaExtract financial data from contexts."""
    for context in contexts:
        if context.get('extracted_financial_data'):
            return context['extracted_financial_data']
    return {}

def get_reference(context: Dict) -> str:
    """Generate a reference string for the context."""
    ref_parts = []
    
    if context.get("page_number"):
        ref_parts.append(f"p{context['page_number']}")
    if context.get("section_type"):
        ref_parts.append(context["section_type"])
    if context.get("file_name"):
        ref_parts.append(context["file_name"])
    
    return " | ".join(ref_parts) if ref_parts else "Unknown"

def extract_financial_highlights(contexts: List[Dict]) -> List[str]:
    """Extract key financial numbers and metrics."""
    highlights = []
    
    for context in contexts:
        text = context.get("text", "")
        
        # Look for currency amounts
        currency_matches = re.findall(r'\$([\d,]+\.?\d*)\s*(million|billion|thousand)?', text, re.IGNORECASE)
        for amount, unit in currency_matches:
            try:
                value = float(amount.replace(',', ''))
                if unit:
                    if unit.lower() == 'million':
                        value *= 1000000
                    elif unit.lower() == 'billion':
                        value *= 1000000000
                    elif unit.lower() == 'thousand':
                        value *= 1000
                highlights.append(f"${value:,.0f}")
            except ValueError:
                continue
        
        # Look for percentages
        percent_matches = re.findall(r'([\d,]+\.?\d*)\s*%', text)
        for percent in percent_matches:
            try:
                value = float(percent.replace(',', ''))
                highlights.append(f"{value}%")
            except ValueError:
                continue
    
    # Return unique highlights, limited to reasonable number
    return list(set(highlights))[:8]
