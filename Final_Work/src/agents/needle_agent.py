from typing import List, Dict
import re

def run_needle(query: str, contexts: List[Dict]) -> str:
    """Find specific information or answer precise questions."""
    
    # Extract key terms from query
    query_terms = extract_query_terms(query)
    
    # Score contexts based on relevance
    scored = []
    for c in contexts:
        text = (c.get("text") or "").lower()
        score = calculate_relevance_score(query_terms, text, c)
        
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
            ref = get_reference(match)
            answer_parts.append(f"Match {i+1} ({ref}):")
            
            # Extract relevant text snippet
            snippet = extract_relevant_snippet(query_terms, match.get("text", ""))
            answer_parts.append(f"{snippet}")
            answer_parts.append("")
    
    return "\n".join(answer_parts)

def run_needle_with_hybrid_retrieval(query: str, hybrid_retriever, k: int = 10) -> str:
    """Find specific information using hybrid retrieval (Pinecone + TF-IDF + Reranking)."""
    
    try:
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
        query_terms = extract_query_terms(query)
        
        # Generate answer from retrieved hits
        answer_parts = []
        answer_parts.append(f"🔍 Found {len(hits)} relevant chunks for: '{query}'")
        answer_parts.append("")
        
        for i, hit in enumerate(hits[:5]):  # Show top 5 results
            ref = get_reference(hit)
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
            
            snippet = extract_relevant_snippet(query_terms, text)
            
            answer_parts.append(f"Match {i+1} (Score: {score:.3f}, {ref}):")
            answer_parts.append(f"{snippet}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
        
    except Exception as e:
        return f"Error in hybrid retrieval: {str(e)}"

def extract_query_terms(query: str) -> List[str]:
    """Extract key terms from the query with Hebrew-English mapping."""
    # Remove common words and extract meaningful terms
    stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    
    terms = re.findall(r'\b\w+\b', query.lower())
    meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
    
    # Add Hebrew equivalents for financial terms
    hebrew_mappings = {
        'revenue': ['הכנסה', 'רווח', 'תקבול', 'הכנסות'],
        'income': ['הכנסה', 'רווח', 'תקבול', 'הכנסות'],
        'profit': ['רווח', 'הכנסה', 'רווחים'],
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
        'indicators': ['מדדים', 'מחוונים', 'סמנים']
    }
    
    # Expand terms with Hebrew equivalents
    expanded_terms = []
    for term in meaningful_terms:
        expanded_terms.append(term)
        if term in hebrew_mappings:
            expanded_terms.extend(hebrew_mappings[term])
    
    return expanded_terms

def calculate_relevance_score(query_terms: List[str], text: str, context: Dict) -> float:
    """Calculate relevance score for a context."""
    score = 0.0
    
    # Exact term matches
    for term in query_terms:
        if term in text:
            score += 1.0
    
    # Phrase matches (consecutive terms)
    if len(query_terms) > 1:
        for i in range(len(query_terms) - 1):
            phrase = f"{query_terms[i]} {query_terms[i+1]}"
            if phrase in text:
                score += 2.0
    
    # Boost for financial terms (English and Hebrew)
    financial_terms = [
        "revenue", "profit", "loss", "income", "expense", "asset", "liability", "equity", 
        "million", "quarter", "q1", "q2", "q3", "q4",
        "הכנסה", "רווח", "הפסד", "תקבול", "נכסים", "התחייבויות", "מיליון", "רבעון"
    ]
    
    # Boost for any Hebrew terms (since they're likely relevant in Hebrew documents)
    hebrew_terms = [term for term in query_terms if any(c in 'אבגדהוזחטסעפצקרשת' for c in term)]
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

def extract_relevant_snippet(query_terms: List[str], text: str, max_length: int = 400) -> str:
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
