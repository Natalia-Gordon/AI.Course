from typing import List, Dict
import re
from datetime import datetime

def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    """Extract keywords with financial domain focus."""
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    
    # Financial domain stop words
    stop = set("""
        a an the and or of to for in is are on with by from as at be been was were 
        it this that these those not no yes if then else than we you they i vs etc per
        million billion thousand revenue income profit loss financial quarter year
        company business market share price stock bond investment asset liability
    """.split())
    
    freq = {}
    for t in tokens:
        if t in stop or len(t) < 3:
            continue
        freq[t] = freq.get(t, 0) + 1
    
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def summarize_text(text: str, max_words: int = 60) -> str:
    """Create a concise summary of the text."""
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Try to find a good breaking point (end of sentence)
    summary = " ".join(words[:max_words])
    if "." in summary:
        last_sentence = summary.rsplit(".", 1)[0] + "."
        return last_sentence
    
    return summary + "..."

def extract_financial_metrics(text: str) -> Dict[str, float]:
    """Extract financial metrics from text."""
    metrics = {}
    
    # Currency patterns
    currency_patterns = [
        r'\$([\d,]+\.?\d*)',
        r'([\d,]+\.?\d*)\s*million',
        r'([\d,]+\.?\d*)\s*billion',
        r'([\d,]+\.?\d*)\s*thousand'
    ]
    
    for pattern in currency_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                try:
                    value = float(match.replace(',', ''))
                    if 'million' in pattern:
                        value *= 1000000
                    elif 'billion' in pattern:
                        value *= 1000000000
                    elif 'thousand' in pattern:
                        value *= 1000
                    metrics[f"currency_{len(metrics)}"] = value
                except ValueError:
                    continue
    
    # Percentage patterns
    percent_patterns = [
        r'([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*percent'
    ]
    
    for pattern in percent_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match.replace(',', ''))
                metrics[f"percentage_{len(metrics)}"] = value
            except ValueError:
                continue
    
    return metrics

def extract_dates(text: str) -> List[str]:
    """Extract dates from text."""
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        r'\bQ[1-4]\s+\d{4}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return list(set(dates))

def analyze_document_structure(text: str) -> Dict[str, any]:
    """Analyze the structure and characteristics of the document."""
    analysis = {
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[.!?]+', text)),
        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
        "has_numbers": bool(re.search(r'\d', text)),
        "has_currency": bool(re.search(r'\$', text)),
        "has_percentages": bool(re.search(r'\d+%', text)),
        "has_dates": bool(re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text)),
        "financial_keywords": extract_keywords(text, top_k=5),
        "metrics": extract_financial_metrics(text),
        "dates": extract_dates(text)
    }
    
    return analysis
