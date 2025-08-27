import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .metadata import extract_financial_metrics, extract_dates

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
    
    def extract_incident_info(self, text: str) -> Tuple[Optional[str], Optional[datetime]]:
        """Extract incident type and date if present."""
        # Incident type patterns
        incident_patterns = [
            r'(?:תקרית|אירוע|בעיה|תקלה|כשל|שגיאה)',
            r'(?:הפרה|עבירה|תביעה|תלונה)',
            r'(?:תאונה|נזק|אובדן|גניבה)',
        ]
        
        incident_type = None
        for pattern in incident_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                incident_type = "security_incident"  # Default type
                break
        
        # Try to extract incident date
        dates = self.extract_dates(text)
        incident_date = None
        
        if dates:
            # Try to parse the first date found
            try:
                # Simple date parsing - you might want to use a more robust parser
                for date_str in dates:
                    if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                        incident_date = datetime.strptime(date_str, '%d/%m/%Y')
                        break
            except ValueError:
                pass
        
        return incident_type, incident_date
