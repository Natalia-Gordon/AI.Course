from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class SectionType(str, Enum):
    """Section types for document chunks."""
    SUMMARY = "Summary"
    TIMELINE = "Timeline"
    TABLE = "Table"
    FIGURE = "Figure"
    ANALYSIS = "Analysis"
    CONCLUSION = "Conclusion"
    OWNERSHIP = "Ownership"  # New section type for ownership information
    SHAREHOLDERS = "Shareholders"  # New section type for shareholder details
    CORPORATE_GOVERNANCE = "Corporate_Governance"  # New section type for governance info

class Language(str, Enum):
    """Supported languages."""
    HEBREW = "he"
    ENGLISH = "en"
    ARABIC = "ar"

class Audience(str, Enum):
    """Document audience types."""
    INTERNAL = "internal"
    PUBLIC = "public"

class Permission(str, Enum):
    """User permission levels."""
    USER = "role:user"
    ANALYST = "role:analyst"
    ADMIN = "role:admin"

class Collection(str, Enum):
    """Document collection types."""
    PRODUCT_DOCS = "product_docs"
    POLICIES = "policies"
    KNOWLEDGE_BASE = "knowledge_base"
    TICKETS = "tickets"

class CriticalEntity(str, Enum):
    """Critical entity types for extraction."""
    ORGANIZATION = "ORG"
    PERSON = "PERSON"
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    ID = "ID"
    KPI = "KPI"

class OwnershipEntity(str, Enum):
    """Ownership entity types for extraction."""
    CONTROLLING_OWNER = "CONTROLLING_OWNER"
    SHAREHOLDER = "SHAREHOLDER"
    MAJORITY_OWNER = "MAJORITY_OWNER"
    MINORITY_OWNER = "MINORITY_OWNER"
    VOTING_RIGHTS_HOLDER = "VOTING_RIGHTS_HOLDER"

class DocumentChunk(BaseModel):
    """Complete metadata schema for document chunks as per project requirements."""
    
    # Core identifiers
    id: str = Field(..., description="Unique chunk identifier")
    file_name: str = Field(..., description="Source document filename")
    client_id: Optional[str] = Field(None, description="Client identifier")
    case_id: Optional[str] = Field(None, description="Case identifier")
    
    # Location and structure
    page_number: int = Field(..., description="Page number in document")
    section_type: SectionType = Field(..., description="Type of section")
    table_id: Optional[str] = Field(None, description="Table identifier if applicable")
    figure_id: Optional[str] = Field(None, description="Figure identifier if applicable")
    row_idx: Optional[int] = Field(None, description="Row index in table")
    col_idx: Optional[int] = Field(None, description="Column index in table")
    
    # Content information
    chunk_index: int = Field(..., description="Sequential chunk index")
    chunk_tokens: int = Field(..., description="Number of tokens in chunk")
    chunk_summary: str = Field(..., description="Brief summary of chunk content")
    text: str = Field(..., description="Full chunk text content")
    
    # Extracted metadata
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    critical_entities: List[CriticalEntity] = Field(default_factory=list, description="Critical entities found")
    
    # NEW: Ownership-specific metadata
    ownership_entities: List[OwnershipEntity] = Field(default_factory=list, description="Ownership entity types found in chunk")
    ownership_percentages: List[float] = Field(default_factory=list, description="Ownership percentages mentioned")
    ownership_companies: List[str] = Field(default_factory=list, description="Company names mentioned in ownership context")
    ownership_dates: List[str] = Field(default_factory=list, description="Dates mentioned in ownership context")
    has_ownership_info: bool = Field(default=False, description="Whether chunk contains ownership information")
    ownership_confidence: float = Field(default=0.0, description="Confidence score for ownership information (0.0-1.0)")
    
    amount_range: Optional[List[float]] = Field(None, description="Amount range [min, max] if applicable")
    
    # Document properties
    language: Language = Field(default=Language.HEBREW, description="Document language")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Access control
    audience: Audience = Field(default=Audience.INTERNAL, description="Target audience")
    permissions: List[Permission] = Field(default_factory=lambda: [Permission.USER], description="Required permissions")
    
    # Classification
    collection: Collection = Field(default=Collection.KNOWLEDGE_BASE, description="Document collection")
    topic: List[str] = Field(default_factory=list, description="Topic tags")
    
    # References
    uri: Optional[str] = Field(None, description="Document URI")
    
    # Quality metrics
    freshness: float = Field(default=1.0, description="Content freshness score (0.0-1.0)")
    priority: float = Field(default=0.5, description="Priority score (0.0-1.0)")

class TableMetadata(BaseModel):
    """Metadata for table structures."""
    table_id: str
    file_name: str
    page_number: int
    section_type: SectionType = SectionType.TABLE
    client_id: Optional[str] = None
    case_id: Optional[str] = None
    
    # Table structure
    rows: int
    columns: int
    headers: List[str]
    
    # Content description
    caption: Optional[str] = None
    description: Optional[str] = None
    
    # File paths
    csv_path: Optional[str] = None
    markdown_path: Optional[str] = None

class FigureMetadata(BaseModel):
    """Metadata for figure structures."""
    figure_id: str
    file_name: str
    page_number: int
    section_type: SectionType = SectionType.FIGURE
    client_id: Optional[str] = None
    case_id: Optional[str] = None
    
    # Figure properties
    caption: Optional[str] = None
    description: Optional[str] = None
    figure_type: str = "chart"  # chart, graph, image, diagram
    
    # File paths
    image_path: Optional[str] = None
    markdown_path: Optional[str] = None

def create_chunk_id(file_name: str, chunk_index: int) -> str:
    """Generate a unique chunk ID."""
    return f"{file_name.replace('.', '_')}_{chunk_index:04d}"

def validate_metadata(chunk: DocumentChunk) -> bool:
    """Validate that chunk metadata meets project requirements."""
    required_fields = [
        'id', 'file_name', 'page_number', 'section_type', 
        'chunk_index', 'chunk_tokens', 'chunk_summary', 'text'
    ]
    
    for field in required_fields:
        if not getattr(chunk, field):
            return False
    
    # Validate chunk size (should be ≤5% of original document)
    if chunk.chunk_tokens <= 0:
        return False
    
    # Validate section type
    if chunk.section_type not in SectionType:
        return False
    
    return True
