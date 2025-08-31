from typing import List, Dict
from pathlib import Path
import re
from datetime import datetime
from .metadata import summarize_text, extract_keywords, analyze_document_structure
from .entity_extractor import EntityExtractor
# TableProcessor removed - table processing now handled by table_enhancer.py
from utils.schema import DocumentChunk, SectionType, Language, Audience, Permission, Collection, CriticalEntity

def _split_by_headings(text: str) -> List[str]:
    """Split text by headings and natural breaks."""
    sections, current = [], []
    
    for line in text.splitlines():
        line = line.strip()
        # Detect headings (all caps, numbers, or specific patterns)
        if (line.isupper() and len(line) > 3) or \
           re.match(r'^\d+\.', line) or \
           re.match(r'^[A-Z][a-z]+:', line) or \
           line in ['Summary', 'Financial Results', 'Risk Factors', 'Outlook', 'תקציר', 'תוצאות פיננסיות']:
            if current:
                sections.append("\n".join(current).strip())
                current = []
        current.append(line)
    
    if current:
        sections.append("\n".join(current).strip())
    
    return [s for s in sections if s and len(s.split()) > 10]

def chunk_document(file_name: str, blocks: List[Dict], budget_ratio: float = 0.05, max_chunk_tokens: int = 400):
    """Create chunks from document blocks with complete metadata."""
    chunks = []
    chunk_id = 0
    
    # Initialize processors
    entity_extractor = EntityExtractor()
    # TableProcessor removed - table processing now handled by table_enhancer.py
    
    for block in blocks:
        text = block.get("text", "")
        if not text.strip():
            continue
            
        # Split block into sections
        sections = _split_by_headings(text)
        
        for section in sections:
            words = section.split()
            if len(words) < 20:  # Skip very short sections
                continue
                
            # Extract entities and metadata
            entities = entity_extractor.extract_entities(section)
            incident_type, incident_date = entity_extractor.extract_incident_info(section)
            amount_range = entity_extractor.extract_amount_range(section)
            language = entity_extractor.detect_language(section)
            
            # Table processing removed - now handled by table_enhancer.py
            tables = []
            has_tables = False
            
            # Create chunk with complete metadata
            chunk = DocumentChunk(
                id=f"{Path(file_name).stem}_{chunk_id:04d}",
                file_name=file_name,
                client_id=None,  # Will be populated from LlamaExtract company extraction
                case_id=None,  # Financial reports don't have case IDs
                page_number=block.get("page_number", 1),
                section_type=block.get("section_type") if isinstance(block.get("section_type"), SectionType) else SectionType.ANALYSIS,
                table_id=tables[0].get("table_id") if tables else None,
                figure_id=None,
                row_idx=None,
                col_idx=None,
                chunk_index=chunk_id,
                chunk_tokens=len(words),
                chunk_summary=summarize_text(section),
                text=section,
                keywords=entities.get('keywords', [])[:8],
                critical_entities=_map_entities_to_critical(entities),
                incident_type=incident_type,
                incident_date=incident_date,
                amount_range=amount_range,
                language=Language(language) if language in ["he", "en"] else Language.HEBREW,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                audience=Audience.INTERNAL,
                permissions=[Permission.USER],
                collection=Collection.KNOWLEDGE_BASE,
                topic=_extract_topics(section),
                uri=None,
                freshness=1.0,
                priority=0.5
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Check budget (should be ≤5% of original document)
            if len(chunks) * max_chunk_tokens > len(text.split()) * budget_ratio:
                break
    
    return chunks

def _map_entities_to_critical(entities: Dict[str, List[str]]) -> List[CriticalEntity]:
    """Map extracted entities to critical entity types."""
    critical_entities = []
    
    if entities.get('organizations'):
        critical_entities.append(CriticalEntity.ORGANIZATION)
    if entities.get('persons'):
        critical_entities.append(CriticalEntity.PERSON)
    if entities.get('amounts'):
        critical_entities.append(CriticalEntity.AMOUNT)
    if entities.get('dates'):
        critical_entities.append(CriticalEntity.DATE)
    if entities.get('ids'):
        critical_entities.append(CriticalEntity.ID)
    if entities.get('kpis'):
        critical_entities.append(CriticalEntity.KPI)
    
    return critical_entities

def _extract_topics(text: str) -> List[str]:
    """Extract topic tags from text."""
    topics = []
    text_lower = text.lower()
    
    # Financial topics
    if any(term in text_lower for term in ['רווח', 'הכנסה', 'הוצאה', 'revenue', 'income', 'expense']):
        topics.append('financial')
    
    # Security topics
    if any(term in text_lower for term in ['אבטחה', 'security', 'risk', 'סיכון']):
        topics.append('security')
    
    # HR topics
    if any(term in text_lower for term in ['עובדים', 'employees', 'hr', 'human resources']):
        topics.append('hr')
    
    # Legal topics
    if any(term in text_lower for term in ['חוק', 'law', 'legal', 'תביעה']):
        topics.append('legal')
    
    # API topics
    if any(term in text_lower for term in ['api', 'interface', 'ממשק']):
        topics.append('api')
    
    return topics[:3]  # Limit to 3 topics

def create_semantic_chunks(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """Create overlapping semantic chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_tokens - overlap):
        chunk_words = words[i:i + max_tokens]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    
    return chunks

def validate_chunk_budget(chunks: List[DocumentChunk], original_text: str, budget_ratio: float = 0.05) -> bool:
    """Validate that chunks meet the 5% budget requirement."""
    total_chunk_tokens = sum(chunk.chunk_tokens for chunk in chunks)
    original_tokens = len(original_text.split())
    budget_tokens = original_tokens * budget_ratio
    
    return total_chunk_tokens <= budget_tokens
