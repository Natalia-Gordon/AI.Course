import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .parse_pdf import extract_text_blocks
from .chunking import chunk_document
from .metadata import analyze_document_structure
from datetime import datetime
import hashlib
import re # Added for ownership detection

logger = logging.getLogger(__name__)

# Global cache for LlamaExtract agents and extracted data
_LLAMA_AGENT_CACHE = {}
_EXTRACTED_DATA_CACHE = {}

# Try to import LlamaExtract for enhanced extraction
try:
    from llama_cloud_services import LlamaExtract
    from pydantic import BaseModel, Field
    LLAMA_EXTRACT_AVAILABLE = True
    logger.info("✓ LlamaExtract available for enhanced extraction")
except ImportError:
    LLAMA_EXTRACT_AVAILABLE = False
    logger.warning("⚠ LlamaExtract not available, using standard extraction")

class FinancialExtractionSchema(BaseModel):
    """Schema for extracting financial data using LlamaExtract."""
    
    # Company Information
    company_name: str = Field(description="Name of the company or organization")
    report_period: str = Field(description="Reporting period (e.g., Q1 2025, FY 2024)")
    report_date: str = Field(description="Date of the report")
    
    # Financial Metrics
    revenue: Optional[str] = Field(description="Total revenue for the period")
    net_income: Optional[str] = Field(description="Net income or profit/loss")
    assets: Optional[str] = Field(description="Total assets")
    liabilities: Optional[str] = Field(description="Total liabilities")
    
    # Key Performance Indicators
    kpis: List[str] = Field(description="Key performance indicators mentioned")
    
    # Executive Summary
    executive_summary: str = Field(description="Executive summary or highlights")
    
    # Risk Factors
    risk_factors: List[str] = Field(description="Risk factors or challenges mentioned")
    
    # Outlook
    outlook: Optional[str] = Field(description="Future outlook or guidance")
    
    # OWNERSHIP INFORMATION - PRIORITY FIELDS
    controlling_owner: Optional[str] = Field(description="Name of the controlling owner or controlling shareholder")
    ownership_percentage: Optional[str] = Field(description="Ownership percentage of the controlling owner")
    voting_rights_percentage: Optional[str] = Field(description="Voting rights percentage of the controlling owner")
    ownership_date: Optional[str] = Field(description="Date when the ownership information is valid")
    ownership_structure: Optional[str] = Field(description="Description of the ownership structure")
    major_shareholders: List[str] = Field(description="List of major shareholders and their ownership percentages")

class OwnershipExtractionSchema(BaseModel):
    """Schema for extracting ownership information using LlamaExtract."""
    
    # Ownership Structure
    controlling_owner: str = Field(description="Name of the controlling owner")
    ownership_percentage: str = Field(description="Ownership percentage")
    voting_rights_percentage: str = Field(description="Voting rights percentage")
    ownership_date: str = Field(description="Date of ownership information")
    
    # Company Details
    company_name: str = Field(description="Name of the company being owned")
    company_type: str = Field(description="Type of company (e.g., insurance, holding)")
    
    # Additional Ownership Details
    share_capital: Optional[str] = Field(description="Share capital amount")
    issued_shares: Optional[str] = Field(description="Number of issued shares")
    voting_shares: Optional[str] = Field(description="Number of voting shares")
    
    # Context
    source_section: str = Field(description="Section of document where ownership info was found")
    page_number: Optional[int] = Field(description="Page number where ownership info was found")

class DataLoader:
    """Handles loading and processing of financial documents with optional LlamaExtract enhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents_dir = Path(config.get('documents_dir', 'data/documents'))
        self.processed_dir = Path(config.get('processed_dir', 'data/processed'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LlamaExtract if available
        self.llama_extract = None
        self.financial_agent = None
        self._initialize_llama_extract()
    
    def _initialize_llama_extract(self):
        """Initialize LlamaExtract for enhanced financial data extraction with smart caching."""
        if not LLAMA_EXTRACT_AVAILABLE:
            return
        
        try:
            api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
            if not api_key:
                logger.warning("⚠ LLAMA_CLOUD_API_KEY not found")
                return
            
            # Check global cache first
            cache_key = f"llama_agent_{hashlib.md5(api_key.encode()).hexdigest()[:8]}"
            
            if cache_key in _LLAMA_AGENT_CACHE:
                logger.info("♻️  Reusing cached LlamaExtract agent")
                self.llama_extract = _LLAMA_AGENT_CACHE[cache_key]['llama_extract']
                self.financial_agent = _LLAMA_AGENT_CACHE[cache_key]['financial_agent']
                return
            
            # Create new instance if not cached
            self.llama_extract = LlamaExtract()
            
            # Use a consistent agent name for this process
            agent_name = "financial-report-extractor"
            
            try:
                # Try to get existing agent first (most efficient)
                try:
                    self.financial_agent = self.llama_extract.get_agent(agent_name)
                    logger.info(f"♻️  Reusing existing LlamaExtract agent: {agent_name}")
                except Exception:
                    # Agent doesn't exist, create new one
                    self.financial_agent = self.llama_extract.create_agent(
                        name=agent_name,
                        instructions="""You are a financial analyst specializing in extracting key financial metrics, ownership information, and business insights from financial reports. 

PRIORITY FOCUS: OWNERSHIP INFORMATION
- Pay special attention to ownership structure, controlling shareholders, and ownership percentages
- Look for phrases like "בעלת השליטה", "בעלי השליטה", "הטילשה", "שליטה", "בעלות"
- Extract exact ownership percentages and voting rights percentages
- Identify the controlling owner company name and ownership date
- Look for information about major shareholders and their holdings

Extract all information with high accuracy and completeness, with particular emphasis on ownership details."""
                    )
                    logger.info(f"✅ Created new LlamaExtract agent: {agent_name}")
                
                # Cache the agents
                _LLAMA_AGENT_CACHE[cache_key] = {
                    'llama_extract': self.llama_extract,
                    'financial_agent': self.financial_agent
                }
                
                logger.info("✅ LlamaExtract initialization completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error creating LlamaExtract agent: {e}")
                self.financial_agent = None
                
        except Exception as e:
            logger.error(f"❌ Error initializing LlamaExtract: {e}")
            self.llama_extract = None
            self.financial_agent = None

    def detect_ownership_in_chunk(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Detect ownership information in a chunk and return enhanced metadata."""
        ownership_metadata = {
            'has_ownership_info': False,
            'ownership_entities': [],
            'ownership_percentages': [],
            'ownership_companies': [],
            'ownership_dates': [],
            'ownership_confidence': 0.0
        }
        
        try:
            # Ownership patterns to look for
            ownership_patterns = {
                'controlling_owner': [
                    r'בעלת השליטה',
                    r'בעלי השליטה',
                    r'בעלת המניות',
                    r'בעלי המניות',
                    r'בעלת הבעלות',
                    r'בעלי הבעלות'
                ],
                'ownership_percentages': [
                    r'(\d+\.?\d*)%',
                    r'כ\s*(\d+)',
                    r'בכ\s*(\d+)',
                    r'(\d+)\s*אחוז'
                ],
                'company_names': [
                    r'[א-ת]+(?:\s+[א-ת]+)*\s+(?:בע"מ|בע"מ|חברה|עמותה)',
                    r'(?:חברת|עמותת)\s+[א-ת]+(?:\s+[א-ת]+)*',
                    r'[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Ltd|Inc|Corp|Company)',
                ],
                'dates': [
                    r'(\d{1,2})\s*ביוני\s*(\d{4})',
                    r'(\d{1,2})\s*ביוני',
                    r'יוני\s*(\d{4})',
                    r'(\d{4})'
                ]
            }
            
            # Check for ownership patterns
            ownership_score = 0
            max_score = 10
            
            # Check for controlling owner patterns
            for pattern in ownership_patterns['controlling_owner']:
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    ownership_metadata['ownership_entities'].append('CONTROLLING_OWNER')
                    ownership_score += 3
                    break
            
            # Check for ownership percentages
            for pattern in ownership_patterns['ownership_percentages']:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                for match in matches:
                    try:
                        percentage = float(match)
                        if 0 <= percentage <= 100:
                            ownership_metadata['ownership_percentages'].append(percentage)
                            ownership_score += 2
                    except ValueError:
                        continue
            
            # Check for company names
            for pattern in ownership_patterns['company_names']:
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    ownership_metadata['ownership_companies'].append(pattern.strip())
                    ownership_score += 2
            
            # Check for dates
            for pattern in ownership_patterns['dates']:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        date_str = ' '.join(match)
                    else:
                        date_str = str(match)
                    ownership_metadata['ownership_dates'].append(date_str)
                    ownership_score += 1
            
            # Check for ownership-related keywords
            ownership_keywords = [
                'מניות', 'שליטה', 'בעלות', 'החזקה', 'אחזקה', 'הון', 'הצבעה',
                'בעלים', 'בעלת', 'בעלי', 'שליטה', 'בעלות'
            ]
            
            keyword_count = sum(1 for keyword in ownership_keywords if keyword in chunk_text)
            ownership_score += min(keyword_count, 3)
            
            # Determine if chunk has ownership info
            ownership_metadata['has_ownership_info'] = ownership_score >= 3
            ownership_metadata['ownership_confidence'] = min(ownership_score / max_score, 1.0)
            
            # Remove duplicates
            ownership_metadata['ownership_entities'] = list(set(ownership_metadata['ownership_entities']))
            ownership_metadata['ownership_percentages'] = list(set(ownership_metadata['ownership_percentages']))
            ownership_metadata['ownership_companies'] = list(set(ownership_metadata['ownership_companies']))
            ownership_metadata['ownership_dates'] = list(set(ownership_metadata['ownership_dates']))
            
            logger.debug(f"Ownership detection for chunk: score={ownership_score}, has_ownership={ownership_metadata['has_ownership_info']}")
            
        except Exception as e:
            logger.error(f"Error in ownership detection: {e}")
        
        return ownership_metadata

    def extract_ownership_with_llama(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ownership information using LlamaExtract if available."""
        if not self.financial_agent:
            return chunk_metadata
        
        try:
            # Create a focused prompt for ownership extraction
            ownership_prompt = f"""
            Extract ownership information from the following text. Focus on:
            1. Who is the controlling owner?
            2. What percentage do they own?
            3. What are their voting rights?
            4. When did this ownership take effect?
            5. What company is being owned?
            
            Text: {chunk_text[:1000]}
            
            Return the information in a structured format.
            """
            
            # Use LlamaExtract to extract ownership information
            extraction_result = self.financial_agent.extract(
                text=chunk_text,
                schema=OwnershipExtractionSchema,
                prompt=ownership_prompt
            )
            
            if extraction_result and hasattr(extraction_result, 'data'):
                ownership_data = extraction_result.data
                
                # Update chunk metadata with extracted ownership info
                chunk_metadata.update({
                    'extracted_ownership_data': {
                        'controlling_owner': ownership_data.controlling_owner,
                        'ownership_percentage': ownership_data.ownership_percentage,
                        'voting_rights_percentage': ownership_data.voting_rights_percentage,
                        'ownership_date': ownership_data.ownership_date,
                        'company_name': ownership_data.company_name,
                        'company_type': ownership_data.company_type,
                        'share_capital': ownership_data.share_capital,
                        'issued_shares': ownership_data.issued_shares,
                        'voting_shares': ownership_data.voting_shares,
                        'source_section': ownership_data.source_section,
                        'page_number': ownership_data.page_number
                    },
                    'has_ownership_info': True,
                    'ownership_confidence': 0.9  # High confidence for LlamaExtract results
                })
                
                logger.info(f"✅ Extracted ownership data: {ownership_data.controlling_owner} ({ownership_data.ownership_percentage})")
            
        except Exception as e:
            logger.warning(f"⚠️ LlamaExtract ownership extraction failed: {e}")
        
        return chunk_metadata
    
    def is_llama_extract_available(self) -> bool:
        """Check if LlamaExtract is available and working."""
        return self.financial_agent is not None
    
    def get_llama_extract_status(self) -> str:
        """Get the status of LlamaExtract integration."""
        if not LLAMA_EXTRACT_AVAILABLE:
            return "❌ LlamaExtract module not available"
        
        if not os.environ.get('LLAMA_CLOUD_API_KEY'):
            return "❌ LLAMA_CLOUD_API_KEY not found"
        
        if self.financial_agent:
            return f"✅ LlamaExtract working with agent: {self.financial_agent.name}"
        else:
            return "⚠ LlamaExtract initialization failed"
    
    @staticmethod
    def clear_cache():
        """Clear all cached data and agents."""
        global _LLAMA_AGENT_CACHE, _EXTRACTED_DATA_CACHE
        _LLAMA_AGENT_CACHE.clear()
        _EXTRACTED_DATA_CACHE.clear()
        logger.info("🗑️  Cache cleared")
    
    @staticmethod
    def get_cache_stats():
        """Get cache statistics."""
        return {
            'cached_agents': len(_LLAMA_AGENT_CACHE),
            'cached_extractions': len(_EXTRACTED_DATA_CACHE),
            'total_cache_size': len(_LLAMA_AGENT_CACHE) + len(_EXTRACTED_DATA_CACHE)
        }
    
    def extract_financial_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract structured financial data using LlamaExtract with caching."""
        if not self.financial_agent:
            logger.warning("LlamaExtract not available, skipping structured extraction")
            return None
        
        # Check cache first
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        cache_key = f"extracted_data_{file_hash}"
        
        if cache_key in _EXTRACTED_DATA_CACHE:
            logger.info(f"♻️  Using cached financial data for: {Path(file_path).name}")
            return _EXTRACTED_DATA_CACHE[cache_key]
        
        try:
            logger.info(f"Extracting financial data from: {file_path}")
            
            # Use LlamaExtract to extract structured data
            result = self.financial_agent.extract(file_path)
            
            if result and hasattr(result, 'data'):
                logger.info("✓ Financial data extracted successfully")
                # Handle different result structures
                if hasattr(result.data, 'dict'):
                    extracted_data = result.data.dict()
                elif isinstance(result.data, dict):
                    extracted_data = result.data
                else:
                    logger.warning(f"Unexpected result data structure: {type(result.data)}")
                    return None
                
                # Cache the extracted data
                _EXTRACTED_DATA_CACHE[cache_key] = extracted_data
                logger.info(f"💾 Cached financial data for: {Path(file_path).name}")
                
                return extracted_data
            else:
                logger.warning("No financial data extracted")
                return None
                
        except Exception as e:
            logger.error(f"LlamaExtract extraction failed: {e}")
            return None
    
    def enhance_chunks_with_financial_data(self, chunks: List[Dict], financial_data: Dict) -> List[Dict]:
        """Enhance chunks with extracted financial data while preserving existing metadata structure."""
        enhanced_chunks = []
        
        for chunk in chunks:
            enhanced_chunk = chunk.copy()
            
            # Enhance existing metadata fields according to DocumentChunk schema
            
            # 1. Enhance keywords with financial metrics
            if financial_data.get('revenue'):
                enhanced_chunk['keywords'].append(f"Revenue: {financial_data['revenue']}")
            if financial_data.get('net_income'):
                enhanced_chunk['keywords'].append(f"Net Income: {financial_data['net_income']}")
            if financial_data.get('assets'):
                enhanced_chunk['keywords'].append(f"Assets: {financial_data['assets']}")
            if financial_data.get('liabilities'):
                enhanced_chunk['keywords'].append(f"Liabilities: {financial_data['liabilities']}")
            
            # 2. Enhance critical_entities with KPIs and financial data
            if financial_data.get('kpis'):
                enhanced_chunk['critical_entities'].extend([f"KPI:{kpi}" for kpi in financial_data['kpis']])
            
            # Add company and period to critical entities
            if financial_data.get('company_name'):
                enhanced_chunk['critical_entities'].append(f"ORG:{financial_data['company_name']}")
            if financial_data.get('report_period'):
                enhanced_chunk['critical_entities'].append(f"PERIOD:{financial_data['report_period']}")
            if financial_data.get('report_date'):
                enhanced_chunk['critical_entities'].append(f"DATE:{financial_data['report_date']}")
            
            # 3. Enhance chunk_summary with extracted insights
            if financial_data.get('executive_summary'):
                current_summary = enhanced_chunk.get('chunk_summary', '')
                extracted_insight = f"[Extracted: {financial_data['executive_summary'][:100]}...]"
                enhanced_chunk['chunk_summary'] = f"{current_summary} {extracted_insight}".strip()
            
            # 4. Update client_id with extracted company name from LlamaExtract
            if financial_data.get('company_name'):
                enhanced_chunk['client_id'] = financial_data['company_name']
                logger.info(f"✓ Updated client_id to: {financial_data['company_name']}")
            
            # 5. Add extracted data as custom metadata (preserving existing structure)
            enhanced_chunk['extracted_financial_data'] = {
                'company_name': financial_data.get('company_name'),
                'report_period': financial_data.get('report_period'),
                'report_date': financial_data.get('report_date'),
                'revenue': financial_data.get('revenue'),
                'net_income': financial_data.get('net_income'),
                'assets': financial_data.get('assets'),
                'liabilities': financial_data.get('liabilities'),
                'kpis': financial_data.get('kpis', []),
                'risk_factors': financial_data.get('risk_factors', []),
                'outlook': financial_data.get('outlook'),
                # OWNERSHIP INFORMATION
                'controlling_owner': financial_data.get('controlling_owner'),
                'ownership_percentage': financial_data.get('ownership_percentage'),
                'voting_rights_percentage': financial_data.get('voting_rights_percentage'),
                'ownership_date': financial_data.get('ownership_date'),
                'ownership_structure': financial_data.get('ownership_structure'),
                'major_shareholders': financial_data.get('major_shareholders', [])
            }
            
            # 6. Add ownership information to chunk metadata if found
            if financial_data.get('controlling_owner'):
                enhanced_chunk['has_ownership_info'] = True
                enhanced_chunk['ownership_confidence'] = 0.9  # High confidence for LlamaExtract
                
                # Add ownership entities to critical entities
                if financial_data.get('controlling_owner'):
                    enhanced_chunk['critical_entities'].append(f"OWNER:{financial_data['controlling_owner']}")
                if financial_data.get('ownership_percentage'):
                    enhanced_chunk['critical_entities'].append(f"OWNERSHIP:{financial_data['ownership_percentage']}%")
                if financial_data.get('voting_rights_percentage'):
                    enhanced_chunk['critical_entities'].append(f"VOTING:{financial_data['voting_rights_percentage']}%")
            else:
                enhanced_chunk['has_ownership_info'] = False
                enhanced_chunk['ownership_confidence'] = 0.0
            
            # 6. Enhance section_type if executive summary is found
            if financial_data.get('executive_summary') and enhanced_chunk.get('section_type') == 'Analysis':
                enhanced_chunk['section_type'] = 'Summary'  # Upgrade to Summary if executive summary found
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def load_document(self, filename: str) -> List[Dict[str, Any]]:
        """Load and process a single document with optional LlamaExtract enhancement."""
        file_path = self.documents_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        print(f"Processing document: {filename}")
        
        # Extract text blocks
        blocks = extract_text_blocks(str(file_path))
        print(f"Extracted {len(blocks)} text blocks")
        
        # Create chunks
        chunks = chunk_document(
            filename, 
            blocks, 
            budget_ratio=self.config.get('chunking', {}).get('budget_ratio', 0.05),
            max_chunk_tokens=self.config.get('chunking', {}).get('max_chunk_tokens', 400)
        )
        
        print(f"Created {len(chunks)} chunks")
        
        # Add document-level metadata
        for chunk in chunks:
            # Convert DocumentChunk to dict if needed
            if hasattr(chunk, 'dict'):
                chunk_dict = chunk.dict()
            else:
                chunk_dict = chunk
            
            chunk_dict['document_analysis'] = analyze_document_structure(chunk_dict['text'])
            chunks[chunks.index(chunk)] = chunk_dict
        
        # Enhance with LlamaExtract financial data if available
        if self.financial_agent:
            financial_data = self.extract_financial_data(str(file_path))
            if financial_data:
                logger.info(f"📊 LlamaExtract extracted company: {financial_data.get('company_name', 'Not found')}")
                logger.info(f"📊 LlamaExtract extracted period: {financial_data.get('report_period', 'Not found')}")
                
                chunks = self.enhance_chunks_with_financial_data(chunks, financial_data)
                print(f"✓ Enhanced chunks with extracted financial data")
            else:
                logger.warning("⚠ LlamaExtract extraction returned no data")
        else:
            logger.warning("⚠ LlamaExtract not available - client_id will remain None")
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], filename: str):
        """Save processed chunks to JSONL file."""
        output_file = self.processed_dir / f"{Path(filename).stem}_chunks.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Convert datetime objects to ISO format for JSON serialization
                chunk_copy = chunk.copy()
                for key, value in chunk_copy.items():
                    if hasattr(value, 'isoformat'):
                        chunk_copy[key] = value.isoformat()
                    elif hasattr(value, 'value'):  # Handle enum values
                        chunk_copy[key] = value.value
                
                f.write(json.dumps(chunk_copy, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_file}")
        return output_file
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load and process all documents in the documents directory."""
        all_chunks = []
        
        for pdf_file in self.documents_dir.glob('*.pdf'):
            try:
                chunks = self.load_document(pdf_file.name)
                all_chunks.extend(chunks)
                
                # Save individual document chunks
                self.save_chunks(chunks, pdf_file.name)
                
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
        
        # Save combined chunks
        combined_file = self.processed_dir / 'all_chunks.jsonl'
        with open(combined_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def get_processed_chunks(self, filename: str = None) -> List[Dict[str, Any]]:
        """Load previously processed chunks."""
        if filename:
            chunks_file = self.processed_dir / f"{Path(filename).stem}_chunks.jsonl"
        else:
            chunks_file = self.processed_dir / 'all_chunks.jsonl'
        
        if not chunks_file.exists():
            return []
        
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunks.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return chunks
