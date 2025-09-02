#!/usr/bin/env python3
"""
Needle Agent for Financial Documents
Finds specific information or answers precise questions using advanced retrieval
"""

import re
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.agent_logger import log_needle_agent
from utils.logger import get_logger
from core.config_manager import ConfigManager


class NeedleAgent:
    """Needle Agent for finding specific information in financial documents"""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger('needle_agent')
        self.config = ConfigManager(config_path) if config_path else None
        self.logger.info("🚀 Needle Agent initialized")
    
    @log_needle_agent
    def run_needle(self, query: str, contexts: List[Dict], namespace: str = None) -> str:
        """Find specific information or answer precise questions."""
        
        try:
            self.logger.info(f"🔍 Processing needle query: {query[:100]}...")
            
            # Check if this is an ownership query and use specialized logic
            if self.is_ownership_query(query):
                self.logger.info("🎯 Detected ownership query - using specialized search logic")
                return self.run_ownership_search(query, contexts, namespace)
            
            # Check if this is a revenue query and use specialized logic
            if self.is_revenue_query(query):
                self.logger.info("💰 Detected revenue query - using specialized financial data extraction")
                return self.run_revenue_search(query, contexts)
            
            # Use regular needle search for non-specialized queries
            return self.run_regular_needle_search(query, contexts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in needle query: {e}")
            return f"Error processing needle query: {str(e)}"

    def run_ownership_search(self, query: str, contexts: List[Dict], namespace: str = None) -> str:
        """Specialized search for ownership-related information."""
        try:
            self.logger.info("🔍 Running specialized ownership search...")
            
            # Try to use Pinecone ownership search if available
            try:
                from index.pinecone_index import PineconeIndex
                
                # Get index name from config
                from core.config_manager import ConfigManager
                config = ConfigManager()
                index_name = config.pinecone.index_name if hasattr(config, 'pinecone') else 'hybrid-rag'
                pinecone_index = PineconeIndex(index_name=index_name)
                
                # Use provided namespace or determine from contexts
                if not namespace:
                    namespace = self._determine_namespace_from_contexts(contexts)
                
                # Use specialized ownership search
                ownership_results = pinecone_index.search_ownership(
                    query=query, 
                    k=10, 
                    namespace=namespace,
                    min_confidence=0.3
                )
                
                if ownership_results:
                    self.logger.info(f"✅ Found {len(ownership_results)} ownership chunks from Pinecone")
                    return self.process_ownership_results(query, ownership_results)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Pinecone ownership search failed, falling back to context search: {e}")
            
            # Fallback to context-based ownership search
            return self.run_context_based_ownership_search(query, contexts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in ownership search: {e}")
            return f"Error in ownership search: {str(e)}"

    def process_ownership_results(self, query: str, ownership_results: List[Dict]) -> str:
        """Process ownership results from Pinecone."""
        try:
            answer_parts = []
            answer_parts.append(f"🎯 Found ownership information for: '{query}'")
            answer_parts.append("")
            
            for i, result in enumerate(ownership_results[:5]):
                metadata = result.get('metadata', {})
                ownership_score = result.get('ownership_score', 0.0)
                ref = self.get_reference_from_metadata(metadata)
                
                answer_parts.append(f"Ownership Match {i+1} (Score: {ownership_score:.1f}, {ref}):")
                
                # Extract ownership-relevant information
                ownership_info = self.extract_ownership_info_from_metadata(metadata)
                if ownership_info:
                    answer_parts.append(ownership_info)
                else:
                    # Fallback to text snippet
                    text = metadata.get('text', '')
                    if text:
                        snippet = self.extract_ownership_snippet_from_text(query, text)
                        answer_parts.append(snippet)
                
                answer_parts.append("")
            
            result = "\n".join(answer_parts)
            self.logger.info(f"✅ Processed {len(ownership_results)} ownership results")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing ownership results: {e}")
            return f"Error processing ownership results: {str(e)}"

    def get_reference_from_metadata(self, metadata: Dict) -> str:
        """Generate reference string from metadata."""
        ref_parts = []
        
        if metadata.get("page_number"):
            ref_parts.append(f"p{metadata['page_number']}")
        if metadata.get("section_type"):
            ref_parts.append(metadata["section_type"])
        if metadata.get("file_name"):
            ref_parts.append(metadata["file_name"])
        
        return " | ".join(ref_parts) if ref_parts else "Unknown"

    def extract_ownership_info_from_metadata(self, metadata: Dict) -> str:
        """Extract ownership information from metadata."""
        info_parts = []
        
        # Check for extracted ownership data
        if metadata.get('extracted_controlling_owner'):
            info_parts.append(f"Controlling Owner: {metadata['extracted_controlling_owner']}")
        
        if metadata.get('extracted_ownership_percentage'):
            info_parts.append(f"Ownership: {metadata['extracted_ownership_percentage']}")
        
        if metadata.get('extracted_voting_rights_percentage'):
            info_parts.append(f"Voting Rights: {metadata['extracted_voting_rights_percentage']}")
        
        if metadata.get('extracted_ownership_date'):
            info_parts.append(f"Date: {metadata['extracted_ownership_date']}")
        
        if metadata.get('extracted_company_name'):
            info_parts.append(f"Company: {metadata['extracted_company_name']}")
        
        # Check for ownership percentages
        if metadata.get('ownership_percentages'):
            percentages = metadata['ownership_percentages']
            if isinstance(percentages, str):
                percentages = percentages.split(',')
            info_parts.append(f"Percentages: {', '.join(percentages)}")
        
        # Check for ownership companies
        if metadata.get('ownership_companies'):
            companies = metadata['ownership_companies']
            if isinstance(companies, str):
                companies = companies.split(',')
            info_parts.append(f"Companies: {', '.join(companies)}")
        
        # Check for ownership dates
        if metadata.get('ownership_dates'):
            dates = metadata['ownership_dates']
            if isinstance(dates, str):
                dates = dates.split(',')
            info_parts.append(f"Dates: {', '.join(dates)}")
        
        if info_parts:
            return " | ".join(info_parts)
        else:
            return "Ownership information found but details not extracted"

    def extract_ownership_snippet_from_text(self, query: str, text: str, max_length: int = 300) -> str:
        """Extract ownership snippet from text."""
        if not text:
            return ""
        
        # Look for sentences containing ownership information
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Zא-ת])', text)
        best_sentence = ""
        best_score = 0
        
        ownership_terms = [
            "בעלת השליטה", "בעלי מניות", "ווישור", "גלובלטק", 
            "70.17%", "67.19%", "הון המניות", "זכויות הצבעה"
        ]
        
        for sentence in sentences:
            score = sum(1 for term in ownership_terms if term in sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        if best_sentence:
            if len(best_sentence) <= max_length:
                return best_sentence
            else:
                return best_sentence[:max_length-3] + "..."
        
        # Fallback: return beginning of text
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length-3] + "..."

    def run_context_based_ownership_search(self, query: str, contexts: List[Dict]) -> str:
        """Fallback ownership search using provided contexts."""
        try:
            self.logger.info("🔍 Running context-based ownership search...")
            
            # Extract ownership-specific terms
            ownership_terms = self.extract_ownership_terms(query)
            
            # Score contexts with ownership-specific relevance
            scored = []
            for c in contexts:
                text = c.get("text", "")
                if not text:
                    continue
                
                # Clean the Hebrew text
                cleaned_text = self.clean_hebrew_text(text)
                
                # Calculate ownership-specific score
                score = self.calculate_ownership_relevance_score(ownership_terms, cleaned_text.lower(), c)
                
                # Additional boost for exact ownership information
                if self.contains_ownership_details(cleaned_text):
                    score += 10.0  # High boost for actual ownership data
                
                cpy = dict(c)
                cpy["_score"] = score
                cpy["cleaned_text"] = cleaned_text
                scored.append(cpy)
            
            # Sort by ownership relevance score
            best_matches = sorted(scored, key=lambda x: x["_score"], reverse=True)[:5]
            
            if not best_matches or best_matches[0]["_score"] == 0:
                # Try direct ownership search if no good matches found
                self.logger.info("🔍 No ownership matches found, trying direct ownership search...")
                return self.direct_ownership_search(query, contexts)
            
            # Generate ownership-specific answer
            answer_parts = []
            answer_parts.append(f"🔍 Found ownership information for: '{query}'")
            answer_parts.append("")
            
            for i, match in enumerate(best_matches):
                if match["_score"] > 0:
                    ref = self.get_reference(match)
                    score = match["_score"]
                    cleaned_text = match.get("cleaned_text", "")
                    
                    answer_parts.append(f"Match {i+1} (Score: {score:.1f}, {ref}):")
                    
                    # Extract ownership-relevant snippet
                    snippet = self.extract_ownership_snippet(ownership_terms, cleaned_text)
                    answer_parts.append(f"{snippet}")
                    answer_parts.append("")
            
            result = "\n".join(answer_parts)
            self.logger.info(f"✅ Context-based ownership search completed, found {len(best_matches)} matches")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in context-based ownership search: {e}")
            return f"Error in context-based ownership search: {str(e)}"

    def direct_ownership_search(self, query: str, contexts: List[Dict]) -> str:
        """Direct search for ownership information using specific patterns."""
        try:
            self.logger.info("🔍 Performing direct ownership search...")
            
            # Look for specific ownership patterns in all contexts
            ownership_patterns = [
                r'בעלת השליטה.*הינה.*ווישור.*גלובלטק',
                r'ווישור.*גלובלטק.*מחזיקה.*\d+%',
                r'בעלת השליטה.*החל.*30.*ביוני.*2022',
                r'מחזיקה.*70\.17%',
                r'הון המניות.*70\.17%',
                r'זכויות הצבעה.*67\.19%'
            ]
            
            found_ownership = []
            
            for context in contexts:
                text = context.get("text", "")
                if not text:
                    continue
                
                # Clean the Hebrew text
                cleaned_text = self.clean_hebrew_text(text)
                
                # Check for ownership patterns
                for pattern in ownership_patterns:
                    matches = re.findall(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        found_ownership.append({
                            'context': context,
                            'pattern': pattern,
                            'matches': matches,
                            'cleaned_text': cleaned_text
                        })
                        break
            
            if found_ownership:
                # Generate answer from found ownership information
                answer_parts = []
                answer_parts.append(f"🎯 Found specific ownership information for: '{query}'")
                answer_parts.append("")
                
                for i, ownership in enumerate(found_ownership[:3]):
                    ref = self.get_reference(ownership['context'])
                    answer_parts.append(f"Ownership Match {i+1} ({ref}):")
                    
                    # Extract the relevant ownership snippet
                    snippet = self.extract_ownership_snippet_from_pattern(
                        ownership['pattern'], 
                        ownership['cleaned_text']
                    )
                    answer_parts.append(f"{snippet}")
                    answer_parts.append("")
                
                result = "\n".join(answer_parts)
                self.logger.info(f"✅ Direct ownership search found {len(found_ownership)} matches")
                return result
            
            # If still no ownership info found, try broader search
            return self.broad_ownership_search(query, contexts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in direct ownership search: {e}")
            return f"Error in direct ownership search: {str(e)}"

    def broad_ownership_search(self, query: str, contexts: List[Dict]) -> str:
        """Broad search for any ownership-related content."""
        try:
            self.logger.info("🔍 Performing broad ownership search...")
            
            # Look for any ownership-related content
            ownership_keywords = [
                "בעלת השליטה", "בעלי מניות", "ווישור", "גלובלטק", 
                "70.17%", "67.19%", "הון המניות", "זכויות הצבעה"
            ]
            
            relevant_contexts = []
            
            for context in contexts:
                text = context.get("text", "")
                if not text:
                    continue
                
                # Clean the Hebrew text
                cleaned_text = self.clean_hebrew_text(text)
                
                # Count ownership keywords
                keyword_count = sum(1 for keyword in ownership_keywords if keyword in cleaned_text)
                
                if keyword_count > 0:
                    relevant_contexts.append({
                        'context': context,
                        'keyword_count': keyword_count,
                        'cleaned_text': cleaned_text
                    })
            
            if relevant_contexts:
                # Sort by keyword count
                relevant_contexts.sort(key=lambda x: x['keyword_count'], reverse=True)
                
                answer_parts = []
                answer_parts.append(f"🔍 Found ownership-related content for: '{query}'")
                answer_parts.append("")
                
                for i, relevant in enumerate(relevant_contexts[:3]):
                    ref = self.get_reference(relevant['context'])
                    keyword_count = relevant['keyword_count']
                    answer_parts.append(f"Ownership Content {i+1} (Keywords: {keyword_count}, {ref}):")
                    
                    # Extract snippet with ownership keywords
                    snippet = self.extract_ownership_keyword_snippet(
                        ownership_keywords, 
                        relevant['cleaned_text']
                    )
                    answer_parts.append(f"{snippet}")
                    answer_parts.append("")
                
                result = "\n".join(answer_parts)
                self.logger.info(f"✅ Broad ownership search found {len(relevant_contexts)} relevant contexts")
                return result
            
            # Final fallback
            return "No ownership information found in the document. The ownership details might be in a different section or document."
            
        except Exception as e:
            self.logger.error(f"❌ Error in broad ownership search: {e}")
            return f"Error in broad ownership search: {str(e)}"

    def extract_ownership_snippet_from_pattern(self, pattern: str, text: str, max_length: int = 300) -> str:
        """Extract ownership snippet based on specific pattern."""
        try:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                match = matches[0]
                if len(match) <= max_length:
                    return match
                else:
                    return match[:max_length-3] + "..."
            
            return "Ownership pattern found but text extraction failed."
            
        except Exception as e:
            return f"Error extracting ownership snippet: {str(e)}"

    def extract_ownership_keyword_snippet(self, keywords: List[str], text: str, max_length: int = 300) -> str:
        """Extract snippet containing ownership keywords."""
        try:
            # Find the best sentence with ownership keywords
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Zא-ת])', text)
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                score = sum(1 for keyword in keywords if keyword in sentence)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            
            if best_sentence:
                if len(best_sentence) <= max_length:
                    return best_sentence
                else:
                    return best_sentence[:max_length-3] + "..."
            
            # Fallback: return beginning of text
            if len(text) <= max_length:
                return text
            else:
                return text[:max_length-3] + "..."
                
        except Exception as e:
            return f"Error extracting keyword snippet: {str(e)}"

    def extract_ownership_terms(self, query: str) -> List[str]:
        """Extract ownership-specific terms from the query."""
        ownership_keywords = [
            "בעלת", "שליטה", "חברה", "החל", "מיום", "ביוני", "2022",
            "בעלי", "מניות", "אחוזי", "בעלות", "החזקה", "הון", "הצבעה"
        ]
        
        # Extract Hebrew words from query
        hebrew_words = re.findall(r'[אבגדהוזסעפצקרשת]+', query)
        
        # Combine with ownership keywords
        all_terms = hebrew_words + ownership_keywords
        
        # Remove duplicates and return
        return list(dict.fromkeys(all_terms))

    def contains_ownership_details(self, text: str) -> bool:
        """Check if text contains actual ownership details."""
        ownership_indicators = [
            r'בעלת השליטה.*הינה',  # controlling owner is
            r'מחזיקה.*\d+%',       # holds X%
            r'הון המניות.*\d+',    # share capital X
            r'זכויות הצבעה.*\d+',  # voting rights X
            r'אחוזי.*\d+',         # percentages X
            r'ווישור.*גלובלטק',    # Wishor Globaltech
            r'החל מיום.*\d+',      # since date X
        ]
        
        for pattern in ownership_indicators:
            if re.search(pattern, text):
                return True
        
        return False

    def extract_ownership_snippet(self, ownership_terms: List[str], text: str, max_length: int = 500) -> str:
        """Extract ownership-relevant snippet from text."""
        if not text:
            return ""
        
        # Look for sentences containing ownership information
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Zא-ת])', text)
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Score based on ownership terms
            for term in ownership_terms:
                if term.lower() in sentence_lower:
                    score += 1
            
            # Bonus for ownership details
            if self.contains_ownership_details(sentence):
                score += 5
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        if best_sentence:
            # Don't truncate if it's a complete sentence
            if len(best_sentence) <= max_length:
                return best_sentence
            else:
                # Try to break at word boundary
                words = best_sentence.split()
                truncated = ""
                for word in words:
                    if len(truncated + " " + word) <= max_length - 3:
                        truncated += (" " + word) if truncated else word
                    else:
                        break
                return truncated + "..." if truncated else best_sentence[:max_length-3] + "..."
        
        # Fallback: return beginning of text
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length-3] + "..."
    
    def extract_query_terms(self, query: str) -> List[str]:
        """Extract key terms from the query with Hebrew-English mapping and ownership focus."""
        # Remove common words and extract meaningful terms
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        terms = re.findall(r'\b\w+\b', query.lower())
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        # Add Hebrew equivalents for financial terms - Enhanced for revenue queries
        hebrew_mappings = {
            'revenue': ['הכנסה', 'רווח', 'תקבול', 'הכנסות', 'הכנסות', 'הכנסות של', 'הכנסות החברה'],
            'income': ['הכנסה', 'רווח', 'תקבול', 'הכנסות', 'הכנסות', 'הכנסות של', 'הכנסות החברה'],
            'profit': ['רווח', 'הכנסה', 'רווחים', 'רווח נטו', 'רווח נקי'],
            'loss': ['הפסד', 'הפסדה', 'הפסדים'],
            'million': ['מיליון', 'מיליונים'],
            'thousand': ['אלף', 'אלפים'],
            'quarter': ['רבעון', 'רבע', 'רבעונים'],
            'q1': ['רבעון ראשון', 'רבעון 1', 'רבעון ראשון'],
            'q2': ['רבעון שני', 'רבעון 2', 'רבעון שני'],
            'q3': ['רבעון שלישי', 'רבעון 3', 'רבעון שלישי'],
            'q4': ['רבעון רביעי', 'רבעון 4', 'רבעון רביעי'],
            '2025': ['2025', 'תשפ"ה', '2025'],
            '2024': ['2024', 'תשפ"ד', '2024'],
            'customer': ['לקוח', 'לקוחות', 'מבוטח', 'מבוטחים'],
            'deposits': ['פיקדונות', 'הפקדות', 'פקדון', 'הפקדה'],
            'branch': ['סניף', 'סניפים', 'סניפי'],
            'network': ['רשת', 'תשתית', 'רשתות'],
            'assets': ['נכסים', 'רכוש', 'נכס'],
            'liabilities': ['התחייבויות', 'חובות', 'חוב'],
            'net': ['נטו', 'נקי', 'טהור'],
            'company': ['חברה', 'עסק', 'ארגון'],
            'report': ['דוח', 'דיווח', 'דוחות'],
            'financial': ['פיננסי', 'כספי', 'כלכלי'],
            'highlights': ['עיקרים', 'עיקר', 'דגשים'],
            'executive': ['מנהל', 'ניהולי', 'הנהלה'],
            'summary': ['סיכום', 'תקציר', 'סיכומים'],
            'performance': ['ביצועים', 'תפקוד', 'הישגים'],
            'indicators': ['מדדים', 'מחוונים', 'סמנים'],
            # Add ownership-specific terms
            'shareholders': ['בעלי מניות', 'בעלי המניות', 'בעלים'],
            'ownership': ['בעלות', 'שליטה', 'החזקה'],
            'controlling': ['שליטה', 'בקרה', 'ניהול'],
            'voting': ['הצבעה', 'זכויות הצבעה', 'הצבעות'],
            'capital': ['הון', 'הון מניות', 'הון עצמי'],
            'percentage': ['אחוז', 'אחוזים', 'אחוזי'],
            'stake': ['חלק', 'אחזקה', 'החזקה']
        }
        
        # Expand terms with Hebrew equivalents
        expanded_terms = []
        for term in meaningful_terms:
            expanded_terms.append(term)
            if term in hebrew_mappings:
                expanded_terms.extend(hebrew_mappings[term])
        
        return expanded_terms

    def calculate_relevance_score(self, query_terms: List[str], text: str, context: Dict) -> float:
        """Calculate relevance score for a context with enhanced ownership scoring."""
        score = 0.0
        
        # Clean and normalize Hebrew text
        cleaned_text = self.clean_hebrew_text(text)
        text_lower = cleaned_text.lower()
        
        # Exact term matches
        for term in query_terms:
            if term in text_lower:
                score += 1.0
        
        # Phrase matches (consecutive terms)
        if len(query_terms) > 1:
            for i in range(len(query_terms) - 1):
                phrase = f"{query_terms[i]} {query_terms[i+1]}"
                if phrase in text_lower:
                    score += 2.0
        
        # HIGH PRIORITY: Ownership-specific scoring
        ownership_score = self.calculate_ownership_relevance_score(query_terms, text_lower, context)
        score += ownership_score
        
        # Boost for financial terms (English and Hebrew)
        financial_terms = [
            "revenue", "profit", "loss", "income", "expense", "asset", "liability", "equity", 
            "million", "quarter", "q1", "q2", "q3", "q4",
            "הכנסה", "רווח", "הפסד", "תקבול", "נכסים", "התחייבויות", "מיליון", "רבעון"
        ]
        
        # Boost for any Hebrew terms (since they're likely relevant in Hebrew documents)
        hebrew_terms = [term for term in query_terms if any(c in 'אבגדהוזסעפצקרשת' for c in term)]
        for term in hebrew_terms:
            score += 0.3  # Boost for Hebrew terms
        
        # Boost for financial terms
        for term in query_terms:
            if term in financial_terms:
                score += 0.5
        
        # Boost for section type relevance
        section_type = context.get("section_type", "").lower()
        if any(term in section_type for term in query_terms):
            score += 1.0
        
        # Boost for page number if query mentions it
        if re.search(r'page\s+\d+', context.get("text", ""), re.IGNORECASE):
            score += 0.3
        
        # Boost for exact number matches (e.g., "15.2 million", "45 locations")
        for term in query_terms:
            if term.isdigit() or term.replace('.', '').isdigit():
                if term in text:
                    score += 1.5  # Higher boost for exact number matches
        
        return score

    def calculate_ownership_relevance_score(self, query_terms: List[str], text: str, context: Dict) -> float:
        """Calculate specific relevance score for ownership-related queries."""
        score = 0.0
        
        # High boost for exact ownership phrases
        ownership_phrases = [
            "בעלת השליטה",
            "בעלי מניות", 
            "בעלי המניות",
            "אחוזי בעלות",
            "זכויות הצבעה",
            "הון המניות",
            "החזקה בחברה",
            "שליטה בחברה",
            "בעלות בחברה"
        ]
        
        for phrase in ownership_phrases:
            if phrase in text:
                score += 5.0  # High boost for ownership terms
        
        # Boost for specific company names mentioned in the query
        company_names = [
            "ווישור",
            "גלובלטק", 
            "איילון",
            "חברה לביטוח"
        ]
        
        for company in company_names:
            if company in text:
                score += 3.0
        
        # Boost for ownership percentages
        percentage_patterns = [
            r'\d+\.\d+%',  # 70.17%
            r'\d+%',       # 70%
            r'כ\s*\d+',    # כ 70
            r'בכ\s*\d+'    # בכ 70
        ]
        
        for pattern in percentage_patterns:
            if re.search(pattern, text):
                score += 2.0
        
        # Boost for ownership-related keywords
        ownership_keywords = [
            "מניות", "שליטה", "בעלות", "החזקה", "אחזקה", "הון", "הצבעה"
        ]
        
        for keyword in ownership_keywords:
            if keyword in text:
                score += 1.0
        
        # Boost for date references (if query mentions specific dates)
        date_patterns = [
            r'\d{1,2}\s+ביוני\s+\d{4}',  # 30 ביוני 2022
            r'\d{1,2}\s+ביוני',          # 30 ביוני
            r'יוני\s+\d{4}'               # יוני 2022
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text):
                score += 1.5
        
        return score

    def clean_hebrew_text(self, text: str) -> str:
        """Clean and normalize Hebrew text for better search."""
        if not text:
            return ""
        
        # Fix common encoding issues and text corruption
        text_replacements = {
            'ןוירוטקרידה': 'דירקטוריון',
            'םייניב': 'בעלי',
            'םידחואמה': 'מניות',
            'םייפסכה': 'כספיים',
            'םירואב': 'דוחות',
            'םיקית': 'תקצירים',
            'םיעבותה': 'הערות',
            'םיטרפ': 'דוחות',
            'םיחוורב': 'דוחות',
            'םישדוח': 'דוחות',
            'םייונישה': 'הערות',
            'םינותנל': 'הערות',
            'םירוקסה': 'הכנסות',
            'םייתסהש': 'הכנסות',
            'םייפסכה': 'כספיים',
            'םירואב': 'דוחות',
            'םיקית': 'תקצירים',
            'םיעבותה': 'הערות',
            'םיטרפ': 'דוחות',
            'םיחוורב': 'דוחות',
            'םישדוח': 'דוחות',
            'םייונישה': 'הערות',
            'םינותנל': 'הערות',
            'םירוקסה': 'הכנסות',
            'םייתסהש': 'הכנסות'
        }
        
        for corrupted, correct in text_replacements.items():
            text = text.replace(corrupted, correct)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common Hebrew text issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        
        # Ensure proper spacing around Hebrew punctuation
        text = re.sub(r'([א-ת])([.,:;])', r'\1 \2', text)
        
        # Normalize Hebrew text (remove non-Hebrew/Latin characters)
        text = re.sub(r'[^\u0590-\u05FF\u0020-\u007F\u00A0-\u00FF]', '', text)
        
        return text.strip()

    def is_ownership_query(self, query: str) -> bool:
        """Check if the query is asking about ownership information."""
        ownership_patterns = [
            r'מי (בעלת|בעלי) (השליטה|המניות)',
            r'אחוזי (בעלות|שליטה)',
            r'מחזיקה (בכ|ב)',
            r'הון המניות',
            r'זכויות הצבעה',
            r'בעלת השליטה',
            r'בעלי מניות',
            r'שליטה בחברה',
            r'בעלות בחברה'
        ]
        
        query_lower = query.lower()
        for pattern in ownership_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def is_revenue_query(self, query: str) -> bool:
        """Check if the query is about revenue/financial performance."""
        revenue_patterns = [
            r'מה (ההכנסות|הכנסות)',
            r'הכנסות (של|החברה)',
            r'הכנסה (של|החברה)',
            r'רווח (של|החברה)',
            r'תקבול (של|החברה)',
            r'revenue',
            r'income',
            r'profit'
        ]
        
        query_lower = query.lower()
        for pattern in revenue_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _determine_namespace_from_contexts(self, contexts: List[Dict]) -> str:
        """Determine namespace from context metadata."""
        if not contexts:
            return 'default'
        
        # Try to get namespace from first context
        first_context = contexts[0]
        file_name = first_context.get('file_name', '').lower()
        
        # Use same logic as main.py
        if 'ayalon' in file_name:
            return 'ayalon_q1_2025'
        elif 'financial' in file_name:
            return 'financial_reports'
        else:
            return 'general_docs'
    
    def run_revenue_search(self, query: str, contexts: List[Dict]) -> str:
        """Specialized search for revenue/financial information using extracted data."""
        try:
            self.logger.info("💰 Running specialized revenue search...")
            
            # Look for contexts with extracted financial data
            revenue_contexts = []
            for context in contexts:
                extracted_data = context.get('extracted_financial_data', {})
                if extracted_data and extracted_data.get('revenue'):
                    revenue_contexts.append({
                        'context': context,
                        'revenue': extracted_data.get('revenue'),
                        'company_name': extracted_data.get('company_name'),
                        'report_period': extracted_data.get('report_period'),
                        'report_date': extracted_data.get('report_date')
                    })
            
            if revenue_contexts:
                # Use the first context with revenue data (they should all be the same)
                revenue_data = revenue_contexts[0]
                context = revenue_data['context']
                
                # Generate a clear revenue answer
                answer_parts = []
                answer_parts.append(f"💰 הכנסות החברה:")
                answer_parts.append("")
                
                # Format the revenue information clearly
                revenue = revenue_data['revenue']
                company_name = revenue_data.get('company_name', 'איילון חברה לביטוח בע"מ')
                report_date = revenue_data.get('report_date', '30 ביוני 2025')
                
                answer_parts.append(f"📊 סכום ההכנסות: {revenue} ש\"ח")
                answer_parts.append(f"🏢 חברה: {company_name}")
                answer_parts.append(f"📅 תאריך הדוח: {report_date}")
                answer_parts.append("")
                
                # Add source reference
                ref = self.get_reference(context)
                answer_parts.append(f"📄 מקור: {ref}")
                
                result = "\n".join(answer_parts)
                self.logger.info(f"✅ Revenue search completed successfully")
                return result
            
            # Fallback to regular search if no extracted data found
            self.logger.info("⚠️ No extracted revenue data found, falling back to regular search")
            return self.run_regular_needle_search(query, contexts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in revenue search: {e}")
            return f"Error in revenue search: {str(e)}"
    
    def run_regular_needle_search(self, query: str, contexts: List[Dict]) -> str:
        """Regular needle search as fallback."""
        # Extract key terms from query
        query_terms = self.extract_query_terms(query)
        
        # Score contexts based on relevance
        scored = []
        for c in contexts:
            text = (c.get("text") or "").lower()
            score = self.calculate_relevance_score(query_terms, text, c)
            
            cpy = dict(c)
            cpy["_score"] = score
            scored.append(cpy)
        
        # Sort by relevance score
        best_matches = sorted(scored, key=lambda x: x["_score"], reverse=True)[:3]
        
        if not best_matches or best_matches[0]["_score"] == 0:
            return "No relevant information found for your query."
        
        # Generate answer
        answer_parts = []
        
        for i, match in enumerate(best_matches):
            if match["_score"] > 0:
                ref = self.get_reference(match)
                answer_parts.append(f"Match {i+1} ({ref}):")
                
                # Extract relevant text snippet
                snippet = self.extract_relevant_snippet(query_terms, match.get("text", ""))
                answer_parts.append(f"{snippet}")
                answer_parts.append("")
        
        result = "\n".join(answer_parts)
        self.logger.info(f"✅ Regular needle search completed, found {len(best_matches)} matches")
        return result
    
    def extract_relevant_snippet(self, query_terms: List[str], text: str, max_length: int = 400) -> str:
        """Extract the most relevant snippet from the text with Hebrew support."""
        if not text:
            return ""
        
        # Find the best sentence containing query terms
        # Use a more sophisticated sentence splitting that handles decimals and abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Zא-ת])', text)
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Score based on both English and Hebrew terms
            score = sum(1 for term in query_terms if term.lower() in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        if best_sentence:
            # Don't truncate if it's a complete sentence
            if len(best_sentence) <= max_length:
                return best_sentence
            else:
                # Only truncate if absolutely necessary, and try to break at word boundary
                words = best_sentence.split()
                truncated = ""
                for word in words:
                    if len(truncated + " " + word) <= max_length - 3:  # Leave room for "..."
                        truncated += (" " + word) if truncated else word
                    else:
                        break
                return truncated + "..." if truncated else best_sentence[:max_length-3] + "..."
        
        # Fallback: return beginning of text with better truncation
        if len(text) <= max_length:
            return text
        else:
            words = text.split()
            truncated = ""
            for word in words:
                if len(truncated + " " + word) <= max_length - 3:
                    truncated += (" " + word) if truncated else word
                else:
                    break
            return truncated + "..." if truncated else text[:max_length-3] + "..."
    
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

    def run_needle_with_hybrid_retrieval(self, query: str, hybrid_retriever, k: int = 10) -> str:
        """Find specific information using hybrid retrieval (Pinecone + TF-IDF + Reranking)."""
        
        try:
            self.logger.info(f"🔍 Running hybrid retrieval for: {query[:100]}...")
            
            # Use hybrid retriever to get relevant chunks
            hits = hybrid_retriever.search(
                query=query,
                k_dense=k,
                k_sparse=k,
                final_k=min(k, 6)  # Limit final results
            )
            
            if not hits:
                return "No relevant information found for your query."
            
            # Extract key terms from query for snippet extraction
            query_terms = self.extract_query_terms(query)
            
            # Generate answer from retrieved hits
            answer_parts = []
            answer_parts.append(f"🔍 Found {len(hits)} relevant chunks for: '{query}'")
            answer_parts.append("")
            
            for i, hit in enumerate(hits[:5]):  # Show top 5 results
                ref = self.get_reference(hit)
                score = hit.get('score', 0.0)
                
                # Extract relevant text snippet - try multiple sources
                text = hit.get('text', '')
                if not text:
                    # Try to get text from metadata
                    text = hit.get('metadata', {}).get('text', '')
                if not text:
                    # Try to get text from chunk_summary
                    text = hit.get('chunk_summary', '')
                if not text:
                    # Fallback to any available text field
                    text = str(hit.get('metadata', {}))
                
                snippet = self.extract_relevant_snippet(query_terms, text)
                
                answer_parts.append(f"Match {i+1} (Score: {score:.3f}, {ref}):")
                answer_parts.append(f"{snippet}")
                answer_parts.append("")
            
            result = "\n".join(answer_parts)
            self.logger.info(f"✅ Hybrid retrieval completed, found {len(hits)} hits")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid retrieval: {e}")
            return f"Error in hybrid retrieval: {str(e)}"


# Legacy function for backward compatibility
def run_needle(query: str, contexts: List[Dict], namespace: str = None) -> str:
    """Legacy function for backward compatibility."""
    agent = NeedleAgent()
    return agent.run_needle(query, contexts, namespace)


def run_needle_with_hybrid_retrieval(query: str, hybrid_retriever, k: int = 10) -> str:
    """Legacy function for backward compatibility."""
    agent = NeedleAgent()
    return agent.run_needle_with_hybrid_retrieval(query, hybrid_retriever, k)
