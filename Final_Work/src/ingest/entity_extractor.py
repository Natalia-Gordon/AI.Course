import re
import logging
from typing import List, Dict, Any, Optional

from .metadata import extract_financial_metrics, extract_dates

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract critical entities from text for metadata enrichment."""
    
    def __init__(self):
        # Hebrew organization patterns
        self.org_patterns = [
            r'([א-ת]+(?:\s+[א-ת]+)*\s+(?:בע"מ|בע"מ|חברה|עמותה|ארגון))',
            r'(?:חברת|עמותת|ארגון)\s+([א-ת]+(?:\s+[א-ת]+)*)',
            r'([א-ת]+(?:\s+[א-ת]+)*\s+(?:בע"מ|בע"מ))',
        ]
        
        # Hebrew person patterns
        self.person_patterns = [
            r'([א-ת]+\s+[א-ת]+)',  # Basic name pattern
            r'(?:מר|גברת|ד"ר|פרופ)\s+([א-ת]+\s+[א-ת]+)',
        ]
        
        # ID patterns
        self.id_patterns = [
            r'\b(\d{9})\b',  # Israeli ID
            r'\b(\d{10,12})\b',  # General long IDs
            r'\b([A-Z]{2,3}\d{6,8})\b',  # Alphanumeric IDs
        ]
        
        # KPI patterns
        self.kpi_patterns = [
            r'(?:רווח|הכנסה|הוצאה|מכירות|תפוקה|יעילות|איכות)',
            r'(?:ROI|ROE|ROA|EBITDA|P/E|P/B)',
            r'(?:אחוז|פרומיל|בסיס|נקודה)',
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all critical entities from text."""
        entities = {
            'organizations': self.extract_organizations(text),
            'persons': self.extract_persons(text),
            'amounts': self.extract_amounts(text),
            'dates': self.extract_dates(text),
            'ids': self.extract_ids(text),
            'kpis': self.extract_kpis(text),
        }
        
        return entities
    
    def extract_organizations(self, text: str) -> List[str]:
        """Extract organization names."""
        orgs = []
        for pattern in self.org_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            orgs.extend(matches)
        
        # Remove duplicates and clean
        orgs = list(set([org.strip() for org in orgs if len(org.strip()) > 2]))
        return orgs[:10]  # Limit to top 10
    
    def extract_persons(self, text: str) -> List[str]:
        """Extract person names."""
        persons = []
        for pattern in self.person_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            persons.extend(matches)
        
        # Remove duplicates and clean
        persons = list(set([person.strip() for person in persons if len(person.strip()) > 3]))
        return persons[:10]  # Limit to top 10
    
    def extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts and quantities."""
        amounts = []
        
        # Currency amounts
        currency_patterns = [
            r'(\$[\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:שקל|ש"ח|ILS)',
            r'([\d,]+\.?\d*)\s*(?:דולר|USD)',
            r'([\d,]+\.?\d*)\s*(?:אירו|EUR)',
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        # Quantities
        quantity_patterns = [
            r'([\d,]+\.?\d*)\s*(?:יחידות|פריטים|מכירות)',
            r'([\d,]+\.?\d*)\s*(?:אחוז|%)',
        ]
        
        for pattern in quantity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        return list(set(amounts))[:15]
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        dates = extract_dates(text)
        
        # Additional Hebrew date patterns
        hebrew_patterns = [
            r'(\d{1,2}\s+(?:בינואר|פברואר|מרץ|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר)\s+\d{4})',
            r'(\d{1,2}\s+(?:ינ|פב|מרץ|אפר|מאי|יונ|יול|אוג|ספט|אוק|נוב|דצמ)\s+\d{4})',
            r'(?:רבעון|Q)\s*(\d)\s+(\d{4})',
        ]
        
        for pattern in hebrew_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))[:10]
    
    def extract_ids(self, text: str) -> List[str]:
        """Extract various ID numbers."""
        ids = []
        for pattern in self.id_patterns:
            matches = re.findall(pattern, text)
            ids.extend(matches)
        
        return list(set(ids))[:10]
    
    def extract_kpis(self, text: str) -> List[str]:
        """Extract KPI indicators."""
        kpis = []
        for pattern in self.kpi_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            kpis.extend(matches)
        
        return list(set(kpis))[:10]
    
    def extract_ownership_info(self, text: str) -> Dict[str, Any]:
        """Extract ownership information from text using generic patterns."""
        ownership_info = {
            'has_ownership_info': False,
            'ownership_confidence': 0.0,
            'controlling_owner': None,
            'ownership_percentage': None,
            'voting_rights_percentage': None,
            'ownership_date': None,
            'ownership_entities': [],
            'ownership_percentages': [],
            'ownership_companies': [],
            'ownership_dates': []
        }
        
        try:
            # Generic ownership detection patterns
            ownership_indicators = [
                # Hebrew ownership terms
                r'בעלת\s*השליטה', r'בעלי\s*השליטה', r'הטילשה', r'שליטה', r'בעלות',
                r'מחזיק\s*בשליטה', r'שליטה\s*בחברה', r'בעלי\s*מניות',
                # English ownership terms
                r'controlling\s*owner', r'shareholder', r'ownership', r'control',
                r'majority\s*shareholder', r'controlling\s*interest'
            ]
            
            # Count ownership indicators
            indicator_count = 0
            for pattern in ownership_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    indicator_count += 1
            
            if indicator_count >= 2:
                ownership_info['has_ownership_info'] = True
                ownership_info['ownership_confidence'] = 0.7
                
                # Extract company names using generic patterns
                company_patterns = [
                    r'([א-ת]+(?:\s+[א-ת]+)*\s+(?:בע"מ|בע"מ|חברה|עמותה))',
                    r'(?:חברת|עמותת)\s+([א-ת]+(?:\s+[א-ת]+)*)',
                    r'([A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Ltd|Inc|Corp|Company))',
                ]
                
                companies = []
                for pattern in company_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    companies.extend(matches)
                
                if companies:
                    ownership_info['controlling_owner'] = companies[0]
                    ownership_info['ownership_companies'] = companies[:3]
                
                # Extract ownership percentages using generic patterns
                percentage_patterns = [
                    r'(\d+\.?\d*)%\s*(?:מהון\s*המניות|מהמניות|מהון)',
                    r'(\d+\.?\d*)%\s*(?:מזכויות\s*הצבעה|זכויות\s*הצבעה)',
                    r'(\d+\.?\d*)%\s*(?:ownership|shareholding)',
                    r'(\d+\.?\d*)%-כב',  # Reversed Hebrew pattern
                ]
                
                percentages = []
                for pattern in percentage_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    percentages.extend(matches)
                
                if percentages:
                    ownership_info['ownership_percentage'] = percentages[0]
                    ownership_info['ownership_percentages'] = percentages[:3]
                
                # Extract voting rights percentages
                voting_patterns = [
                    r'(\d+\.?\d*)%.*?(?:בדילול\s*מלא|voting\s*rights)',
                    r'(\d+\.?\d*)%.*?לולידב',  # Reversed Hebrew
                ]
                
                for pattern in voting_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        ownership_info['voting_rights_percentage'] = match.group(1)
                        break
                
                # Extract ownership dates using generic patterns
                date_patterns = [
                    r'(\d{4}\s+[א-ת]+\s+\d{1,2})',  # Hebrew date
                    r'(\d{1,2}\s+[א-ת]+\s+\d{4})',  # Hebrew date
                    r'(\d{1,2}/\d{1,2}/\d{4})',     # DD/MM/YYYY
                    r'(\d{4}-\d{1,2}-\d{1,2})',     # YYYY-MM-DD
                ]
                
                dates = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, text)
                    dates.extend(matches)
                
                if dates:
                    ownership_info['ownership_date'] = dates[0]
                    ownership_info['ownership_dates'] = dates[:3]
                
                # Extract ownership entities (organizations mentioned in ownership context)
                ownership_info['ownership_entities'] = self.extract_organizations(text)
            
        except Exception as e:
            logger.error(f"Error in ownership extraction: {e}")
        
        return ownership_info

    def extract_amount_range(self, text: str) -> Optional[List[float]]:
        """Extract amount range [min, max] if present."""
        # Look for range patterns
        range_patterns = [
            r'(\d+(?:,\d+)*)\s*-\s*(\d+(?:,\d+)*)',
            r'מ\s*(\d+(?:,\d+)*)\s*עד\s*(\d+(?:,\d+)*)',
            r'בין\s*(\d+(?:,\d+)*)\s*ל\s*(\d+(?:,\d+)*)',
        ]
        
        for pattern in range_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    min_val = float(matches[0][0].replace(',', ''))
                    max_val = float(matches[0][1].replace(',', ''))
                    return [min_val, max_val]
                except ValueError:
                    continue
        
        return None
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text."""
        hebrew_chars = len(re.findall(r'[א-ת]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if hebrew_chars > english_chars:
            return "he"
        elif english_chars > hebrew_chars:
            return "en"
        else:
            return "mixed"
    

