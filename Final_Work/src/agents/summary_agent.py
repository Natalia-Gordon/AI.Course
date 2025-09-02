#!/usr/bin/env python3
"""
Summary Agent for Financial Documents
Generates comprehensive summaries from retrieved content with LlamaExtract enhancement
"""

import re
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.agent_logger import log_summary_agent
from utils.logger import get_logger
from core.config_manager import ConfigManager


class SummaryAgent:
    """Summary Agent for generating comprehensive summaries from financial documents"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger('summary_agent')
        self.config = ConfigManager(config_path) if config_path else None
        self.logger.info("🚀 Summary Agent initialized")
    
    def clean_hebrew_text(self, text: str) -> str:
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

    def format_hebrew_summary(self, text: str, max_length: int = 300) -> str:
        """Format Hebrew text for summary display."""
        if not text:
            return ""
        
        # Clean the text
        cleaned = self.clean_hebrew_text(text)
        
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
    
    @log_summary_agent
    def run_summary(self, query: str, contexts: List[Dict]) -> str:
        """Generate a comprehensive summary based on retrieved contexts with LlamaExtract enhancement."""
        
        try:
            self.logger.info(f"📝 Processing summary request: {query[:100]}...")
            
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
                    formatted_summ = self.format_hebrew_summary(summ, 300)
                    ref = self.get_reference(c)
                    summary_parts.append(f"• {formatted_summ} ({ref})")
            
            # Enhanced financial highlights using LlamaExtract data
            financial_data = self.extract_llama_extract_financial_data(contexts)
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
            financial_metrics = self.extract_financial_highlights(contexts)
            if financial_metrics:
                summary_parts.append("\nKEY METRICS:")
                for metric in financial_metrics[:5]:
                    summary_parts.append(f"• {metric}")
            
            # AUTOMATIC TABLE INCLUSION - Fetch relevant tables from Pinecone
            table_summary = self.generate_table_summary_from_pinecone(contexts)
            if table_summary:
                summary_parts.append(f"\nTABLE SUMMARIES:")
                summary_parts.append(table_summary)
            
            # Other sections
            other_sections = [s for s in by_section.keys() if s not in ["Summary", "Financial"]]
            for section in other_sections[:2]:  # Top 2 other sections
                summary_parts.append(f"\n{section.upper()}:")
                for c in by_section[section][:2]:
                    summ = c.get("chunk_summary") or c.get("text", "")
                    formatted_summ = self.format_hebrew_summary(summ, 200)
                    ref = self.get_reference(c)
                    summary_parts.append(f"• {formatted_summ} ({ref})")
            
            result = "\n".join(summary_parts)
            self.logger.info(f"✅ Summary generated successfully, {len(summary_parts)} sections")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def extract_llama_extract_financial_data(self, contexts: List[Dict]) -> Dict:
        """Extract LlamaExtract financial data from contexts."""
        for context in contexts:
            if context.get('extracted_financial_data'):
                return context['extracted_financial_data']
        return {}

    def get_reference(self, context: Dict) -> str:
        """Generate a reference string for the context."""
        ref_parts = []
        
        if context.get("page_number"):
            ref_parts.append(f"p{context['page_number']}")
        if context.get("section_type"):
            ref_parts.append(context["section_type"])
        if context.get("file_name"):
            ref_parts.append(context["file_name"])
        
        return " | ".join(ref_parts) if ref_parts else "Unknown"

    def extract_financial_highlights(self, contexts: List[Dict]) -> List[str]:
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

    def generate_table_summary_from_pinecone(self, contexts: List[Dict]) -> str:
        """Automatically fetch and generate table summaries from Pinecone based on context relevance."""
        try:
            # Import PineconeIndex here to avoid circular imports
            from index.pinecone_index import PineconeIndex
            
            # Get index name from config
            from core.config_manager import ConfigManager
            config = ConfigManager()
            index_name = config.pinecone.index_name if hasattr(config, 'pinecone') else 'financial-reports'
            
            # Initialize Pinecone connection
            pinecone_index = PineconeIndex(index_name=index_name)
            namespace = 'ayalon_q1_2025'
            
            # Extract relevant keywords from contexts for table search
            search_terms = self.extract_search_terms_from_contexts(contexts)
            
            # Search for relevant tables
            table_results = []
            for term in search_terms[:3]:  # Use top 3 most relevant terms
                results = pinecone_index.search(term, k=5, namespace=namespace)
                for result in results:
                    metadata = result.get('metadata', {})
                    if metadata.get('section_type') == 'Table':
                        # Check if this table is already in our results
                        table_id = metadata.get('table_id')
                        if not any(t.get('metadata', {}).get('table_id') == table_id for t in table_results):
                            table_results.append(result)
            
            # Limit to top 5 most relevant tables
            table_results = table_results[:5]
            
            if not table_results:
                return ""
            
            # Generate table summaries
            table_summaries = []
            for table in table_results:
                metadata = table.get('metadata', {})
                table_id = metadata.get('table_id', 'Unknown')
                summary = metadata.get('chunk_summary', '')
                page = metadata.get('page_number', 'Unknown')
                
                if summary:
                    # Format table summary
                    formatted_summary = self.format_hebrew_summary(summary, 150)
                    table_summaries.append(f"• {formatted_summary} (Table: {table_id}, Page: {page})")
            
            return "\n".join(table_summaries)
            
        except Exception as e:
            # If there's any error, return empty string (don't break the summary)
            self.logger.warning(f"⚠️ Could not fetch table summaries: {e}")
            return ""

    def extract_search_terms_from_contexts(self, contexts: List[Dict]) -> List[str]:
        """Extract relevant search terms from contexts for table search."""
        search_terms = []
        
        for context in contexts:
            # Extract from text content
            text = context.get("text", "")
            if text:
                # Look for financial terms
                financial_terms = ['revenue', 'income', 'profit', 'assets', 'liabilities', 'הכנסה', 'רווח', 'נכסים', 'התחייבויות']
                for term in financial_terms:
                    if term.lower() in text.lower() and term not in search_terms:
                        search_terms.append(term)
            
            # Extract from keywords
            keywords = context.get("keywords", [])
            if keywords:
                for keyword in keywords[:3]:  # Top 3 keywords
                    if keyword not in search_terms:
                        search_terms.append(str(keyword))
            
            # Extract from table_id if present (for table-related contexts)
            table_id = context.get("table_id")
            if table_id and table_id not in search_terms:
                search_terms.append(str(table_id))
        
        # Return unique terms, prioritizing financial terms
        return list(dict.fromkeys(search_terms))  # Preserves order while removing duplicates


# Legacy function for backward compatibility
def run_summary(query: str, contexts: List[Dict]) -> str:
    """Legacy function for backward compatibility."""
    agent = SummaryAgent()
    return agent.run_summary(query, contexts)
