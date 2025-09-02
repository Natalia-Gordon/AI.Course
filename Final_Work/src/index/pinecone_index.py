import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from utils.schema import DocumentChunk

class PineconeIndex:
    """Pinecone vector database index for dense retrieval."""
    
    def __init__(self, api_key: str = None, region: str = None, cloud: str = None, 
                 index_name: str = None, dimension: int = 1536, metric: str = "cosine"):
        """
        Initialize Pinecone index.
        
        Args:
            api_key: Pinecone API key (from PINECONE_API_KEY env var)
            region: Pinecone region (from PINECONE_REGION env var)
            cloud: Pinecone cloud provider (from PINECONE_CLOUD env var)
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.region = region or os.environ.get("PINECONE_REGION")
        self.cloud = cloud or os.environ.get("PINECONE_CLOUD", "aws")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.region:
            raise ValueError("PINECONE_REGION environment variable is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        # Create or connect to index
        self._setup_index()
    
    def _setup_index(self):
        """Create index if it doesn't exist, or connect to existing one."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating Pinecone index: {self.index_name}")
                
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                print(f"✓ Index '{self.index_name}' created successfully")
            else:
                print(f"✓ Index '{self.index_name}' already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print(f"✓ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"✗ Error setting up Pinecone index: {e}")
            raise
    
    def build_index(self, chunks: List[Dict[str, Any]], namespace: str = None):
        """Build index from document chunks with optional namespace."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        # Use provided namespace or default
        if namespace:
            print(f"📁 Using namespace: {namespace}")
        else:
            namespace = "__default__"
            print(f"📁 Using default namespace: {namespace}")
        
        try:
            print(f"Building Pinecone index with {len(chunks)} chunks...")
            
            # Analyze chunk structure first
            self.analyze_chunk_structure(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, chunk in enumerate(chunks):
                # Handle both DocumentChunk objects and dictionaries
                if hasattr(chunk, 'file_name'):
                    # DocumentChunk object
                    file_name = chunk.file_name
                    page_number = chunk.page_number
                    section_type = chunk.section_type.value if hasattr(chunk.section_type, 'value') else chunk.section_type
                    chunk_index = chunk.chunk_index
                    chunk_summary = chunk.chunk_summary
                    keywords = chunk.keywords[:5] if chunk.keywords else []
                    client_id = chunk.client_id
                    case_id = chunk.case_id
                    table_id = chunk.table_id
                    figure_id = chunk.figure_id
                    language = chunk.language.value if hasattr(chunk.language, 'value') else chunk.language
                    topic = chunk.topic[:3] if chunk.topic else []
                    text = chunk.text[:500] if chunk.text else ""
                else:
                    # Dictionary
                    file_name = chunk.get('file_name', f'chunk_{i}')
                    page_number = chunk.get('page_number', 0)
                    section_type = chunk.get('section_type', 'Analysis')
                    chunk_index = chunk.get('chunk_index', i)
                    chunk_summary = chunk.get('chunk_summary', '')
                    keywords = chunk.get('keywords', [])[:5]
                    client_id = chunk.get('client_id')
                    case_id = chunk.get('case_id')
                    table_id = chunk.get('table_id')
                    figure_id = chunk.get('figure_id')
                    language = chunk.get('language', 'he')
                    topic = chunk.get('topic', [])[:3]
                    text = chunk.get('text', '')[:500]
                
                # Create vector ID
                vector_id = f"{file_name}_{chunk_index:04d}"
                
                # Prepare metadata - filter out None values for Pinecone compatibility
                metadata = {}
                
                # Only add non-None values
                if file_name is not None:
                    metadata['file_name'] = str(file_name)
                if page_number is not None:
                    metadata['page_number'] = int(page_number)
                if section_type is not None:
                    metadata['section_type'] = str(section_type)
                if chunk_index is not None:
                    metadata['chunk_index'] = int(chunk_index)
                if chunk_summary is not None:
                    metadata['chunk_summary'] = str(chunk_summary)
                
                # Handle keywords (ensure it's a string)
                if keywords:
                    if isinstance(keywords, list):
                        metadata['keywords'] = ','.join(str(k) for k in keywords[:5])
                    else:
                        metadata['keywords'] = str(keywords)
                
                # Only add non-None optional fields
                if client_id is not None:
                    metadata['client_id'] = str(client_id)
                if case_id is not None:
                    metadata['case_id'] = str(case_id)
                if table_id is not None:
                    metadata['table_id'] = str(table_id)
                if figure_id is not None:
                    metadata['figure_id'] = str(figure_id)
                if language is not None:
                    metadata['language'] = str(language)
                
                # Handle topics (ensure it's a string)
                if topic:
                    if isinstance(topic, list):
                        metadata['topic'] = ','.join(str(t) for t in topic[:3])
                    else:
                        metadata['topic'] = str(topic)
                
                # Always add text (truncated)
                if text:
                    metadata['text'] = str(text)[:500]
                
                # NEW: Add ownership-specific metadata
                if hasattr(chunk, 'has_ownership_info') and chunk.has_ownership_info:
                    metadata['has_ownership_info'] = True
                    metadata['ownership_confidence'] = float(chunk.ownership_confidence)
                    
                    # Add ownership entities
                    if hasattr(chunk, 'ownership_entities') and chunk.ownership_entities:
                        if isinstance(chunk.ownership_entities, list):
                            metadata['ownership_entities'] = ','.join(str(e) for e in chunk.ownership_entities)
                        else:
                            metadata['ownership_entities'] = str(chunk.ownership_entities)
                    
                    # Add ownership percentages
                    if hasattr(chunk, 'ownership_percentages') and chunk.ownership_percentages:
                        if isinstance(chunk.ownership_percentages, list):
                            metadata['ownership_percentages'] = ','.join(str(p) for p in chunk.ownership_percentages)
                        else:
                            metadata['ownership_percentages'] = str(chunk.ownership_percentages)
                    
                    # Add ownership companies
                    if hasattr(chunk, 'ownership_companies') and chunk.ownership_companies:
                        if isinstance(chunk.ownership_companies, list):
                            metadata['ownership_companies'] = ','.join(str(c) for c in chunk.ownership_companies)
                        else:
                            metadata['ownership_companies'] = str(chunk.ownership_companies)
                    
                    # Add ownership dates
                    if hasattr(chunk, 'ownership_dates') and chunk.ownership_dates:
                        if isinstance(chunk.ownership_dates, list):
                            metadata['ownership_dates'] = ','.join(str(d) for d in chunk.ownership_dates)
                        else:
                            metadata['ownership_dates'] = str(chunk.ownership_dates)
                    
                    # Add extracted ownership data if available
                    if hasattr(chunk, 'extracted_ownership_data') and chunk.extracted_ownership_data:
                        extracted_data = chunk.extracted_ownership_data
                        if isinstance(extracted_data, dict):
                            for key, value in extracted_data.items():
                                if value is not None:
                                    metadata[f'extracted_{key}'] = str(value)
                
                # Handle dictionary chunks with ownership info
                elif isinstance(chunk, dict) and chunk.get('has_ownership_info'):
                    metadata['has_ownership_info'] = True
                    metadata['ownership_confidence'] = float(chunk.get('ownership_confidence', 0.0))
                    
                    # Add ownership entities
                    ownership_entities = chunk.get('ownership_entities', [])
                    if ownership_entities:
                        if isinstance(ownership_entities, list):
                            metadata['ownership_entities'] = ','.join(str(e) for e in ownership_entities)
                        else:
                            metadata['ownership_entities'] = str(ownership_entities)
                    
                    # Add ownership percentages
                    ownership_percentages = chunk.get('ownership_percentages', [])
                    if ownership_percentages:
                        if isinstance(ownership_percentages, list):
                            metadata['ownership_percentages'] = ','.join(str(p) for p in ownership_percentages)
                        else:
                            metadata['ownership_percentages'] = str(ownership_percentages)
                    
                    # Add ownership companies
                    ownership_companies = chunk.get('ownership_companies', [])
                    if ownership_companies:
                        if isinstance(ownership_companies, list):
                            metadata['ownership_companies'] = ','.join(str(c) for c in ownership_companies)
                        else:
                            metadata['ownership_companies'] = str(ownership_companies)
                    
                    # Add ownership dates
                    ownership_dates = chunk.get('ownership_dates', [])
                    if ownership_dates:
                        if isinstance(ownership_dates, list):
                            metadata['ownership_dates'] = ','.join(str(d) for d in ownership_dates)
                        else:
                            metadata['ownership_dates'] = str(ownership_dates)
                    
                    # Add extracted ownership data if available
                    extracted_data = chunk.get('extracted_ownership_data', {})
                    if extracted_data and isinstance(extracted_data, dict):
                        for key, value in extracted_data.items():
                            if value is not None:
                                metadata[f'extracted_{key}'] = str(value)
                
                # Add to vectors list
                vectors.append({
                    'id': vector_id,
                    'values': text,  # Use the extracted text variable
                    'metadata': metadata
                })
            
            # Generate embeddings for the text (using OpenAI or other embedding model)
            print(f"✓ Prepared {len(vectors)} vectors for indexing")
            
            # For now, create mock embeddings (in production, use OpenAI text-embedding-3-small)
            # This is a placeholder - you should implement proper embedding generation
            import numpy as np
            
            # Create mock embeddings (1536 dimensions as specified in config)
            mock_embeddings = []
            for vector in vectors:
                # Generate a deterministic mock embedding based on text content
                text = vector['values']
                # Simple hash-based mock embedding
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.rand(1536).tolist()
                mock_embeddings.append({
                    'id': vector['id'],
                    'values': embedding,
                    'metadata': vector['metadata']
                })
            
            # Upsert vectors to Pinecone
            print(f"Upserting {len(mock_embeddings)} vectors to Pinecone...")
            
            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            successful_upserts = 0
            failed_upserts = 0
            
            for i in range(0, len(mock_embeddings), batch_size):
                batch = mock_embeddings[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(mock_embeddings) + batch_size - 1)//batch_size
                
                try:
                    # Log the first vector's metadata for debugging
                    if batch_num == 1:
                        first_vector = batch[0]
                        print(f"🔍 Sample metadata for batch 1:")
                        print(f"   ID: {first_vector['id']}")
                        print(f"   Metadata keys: {list(first_vector['metadata'].keys())}")
                        print(f"   Sample values: {dict(list(first_vector['metadata'].items())[:3])}")
                    
                    self.index.upsert(vectors=batch, namespace=namespace)
                    successful_upserts += len(batch)
                    print(f"✓ Upserted batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                    
                except Exception as e:
                    failed_upserts += len(batch)
                    print(f"✗ Error upserting batch {batch_num}/{total_batches}: {e}")
                    
                    # Log detailed error for first batch failure
                    if batch_num == 1:
                        print(f"🔍 First batch error details:")
                        print(f"   Batch size: {len(batch)}")
                        print(f"   First vector ID: {batch[0]['id']}")
                        print(f"   First vector metadata keys: {list(batch[0]['metadata'].keys())}")
                        print(f"   First vector metadata values: {batch[0]['metadata']}")
                    continue
            
            print(f"📊 Upsert Summary:")
            print(f"   Successful: {successful_upserts} vectors")
            print(f"   Failed: {failed_upserts} vectors")
            print(f"   Total: {len(mock_embeddings)} vectors")
            
            if successful_upserts > 0:
                print(f"✅ Successfully upserted {successful_upserts} vectors to Pinecone")
            else:
                print(f"❌ No vectors were upserted successfully")
                raise RuntimeError("All vector upserts failed")
            
            # Store metadata for later use
            self.chunk_metadata = {v['id']: v['metadata'] for v in vectors}
            
        except Exception as e:
            print(f"✗ Error building Pinecone index: {e}")
            raise
    
    def search(self, query: str, k: int = 10, namespace: str = None, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors using Pinecone."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # For now, use mock query embedding (in production, use OpenAI)
            # This is a placeholder - you should implement proper query embedding
            import numpy as np
            
            # Generate mock query embedding
            np.random.seed(hash(query) % 2**32)
            query_embedding = np.random.rand(1536).tolist()
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True,
                filter=filters
            )
            
            # Process results
            results = []
            for match in search_results.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            print(f"✓ Found {len(results)} relevant chunks in Pinecone")
            return results
            
        except Exception as e:
            print(f"✗ Error searching Pinecone index: {e}")
            return []

    def search_ownership(self, query: str, k: int = 10, namespace: str = None, 
                        min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Search specifically for ownership-related information."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # Create ownership-specific filter
            ownership_filter = {
                "has_ownership_info": {"$eq": True},
                "ownership_confidence": {"$gte": min_confidence}
            }
            
            # Generate mock query embedding
            import numpy as np
            np.random.seed(hash(query) % 2**32)
            query_embedding = np.random.rand(1536).tolist()
            
            # Search with ownership filter
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k * 2,  # Get more results to filter
                namespace=namespace,
                include_metadata=True,
                filter=ownership_filter
            )
            
            # Process and score results based on ownership relevance
            results = []
            for match in search_results.matches:
                metadata = match.metadata
                
                # Calculate ownership relevance score
                ownership_score = self.calculate_ownership_relevance_score(query, metadata)
                
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'ownership_score': ownership_score,
                    'metadata': metadata
                })
            
            # Sort by ownership relevance score
            results.sort(key=lambda x: x['ownership_score'], reverse=True)
            
            # Return top k results
            top_results = results[:k]
            
            print(f"✓ Found {len(top_results)} ownership-related chunks in Pinecone")
            return top_results
            
        except Exception as e:
            print(f"✗ Error in ownership search: {e}")
            return []

    def calculate_ownership_relevance_score(self, query: str, metadata: Dict[str, Any]) -> float:
        """Calculate relevance score for ownership information."""
        score = 0.0
        
        # Check for ownership confidence
        ownership_confidence = metadata.get('ownership_confidence', 0.0)
        score += ownership_confidence * 5.0  # High weight for confidence
        
        # Check for specific ownership entities
        ownership_entities = metadata.get('ownership_entities', '')
        if ownership_entities:
            if 'CONTROLLING_OWNER' in ownership_entities:
                score += 3.0
        
        # Check for ownership percentages
        ownership_percentages = metadata.get('ownership_percentages', '')
        if ownership_percentages:
            score += 2.0
        
        # Check for company names
        ownership_companies = metadata.get('ownership_companies', '')
        if ownership_companies:
            score += 2.0
        
        # Check for dates
        ownership_dates = metadata.get('ownership_dates', '')
        if ownership_dates:
            score += 1.0
        
        # Check for extracted ownership data
        extracted_keys = [key for key in metadata.keys() if key.startswith('extracted_')]
        if extracted_keys:
            score += len(extracted_keys) * 0.5
        
        return score

    def search_by_ownership_company(self, company_name: str, namespace: str = None, 
                                   k: int = 10) -> List[Dict[str, Any]]:
        """Search for ownership information by specific company name."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # Create company-specific filter
            company_filter = {
                "has_ownership_info": {"$eq": True},
                "$or": [
                    {"ownership_companies": {"$contains": company_name}},
                    {"extracted_company_name": {"$contains": company_name}},
                    {"client_id": {"$contains": company_name}}
                ]
            }
            
            # Generate mock query embedding
            import numpy as np
            np.random.seed(hash(company_name) % 2**32)
            query_embedding = np.random.rand(1536).tolist()
            
            # Search with company filter
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True,
                filter=company_filter
            )
            
            # Process results
            results = []
            for match in search_results.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            print(f"✓ Found {len(results)} ownership chunks for company: {company_name}")
            return results
            
        except Exception as e:
            print(f"✗ Error in company ownership search: {e}")
            return []

    def search_by_ownership_percentage(self, min_percentage: float, max_percentage: float = 100.0,
                                      namespace: str = None, k: int = 10) -> List[Dict[str, Any]]:
        """Search for ownership information by percentage range."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # Create percentage filter
            percentage_filter = {
                "has_ownership_info": {"$eq": True},
                "ownership_percentages": {"$gte": min_percentage, "$lte": max_percentage}
            }
            
            # Generate mock query embedding
            import numpy as np
            np.random.seed(hash(f"{min_percentage}_{max_percentage}") % 2**32)
            query_embedding = np.random.rand(1536).tolist()
            
            # Search with percentage filter
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True,
                filter=percentage_filter
            )
            
            # Process results
            results = []
            for match in search_results.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            print(f"✓ Found {len(results)} ownership chunks with percentage {min_percentage}-{max_percentage}%")
            return results
            
        except Exception as e:
            print(f"✗ Error in percentage ownership search: {e}")
            return []
    
    def delete_index(self):
        """Delete the Pinecone index."""
        if self.index_name in self.pc.list_indexes().names():
            try:
                self.pc.delete_index(self.index_name)
                print(f"✓ Index '{self.index_name}' deleted successfully")
            except Exception as e:
                print(f"✗ Error deleting index: {e}")
    
    def get_index_stats(self):
        """Get index statistics."""
        if not self.index:
            return None
        
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"✗ Error getting index stats: {e}")
            return None
    
    def check_index_content(self):
        """Check if the index actually contains vectors."""
        if not self.index:
            print("❌ Index not initialized")
            return False
        
        try:
            stats = self.index.describe_index_stats()
            total_vector_count = stats.total_vector_count
            namespaces = stats.namespaces
            
            print(f"📊 Index Statistics:")
            print(f"   Total vectors: {total_vector_count}")
            print(f"   Namespaces: {list(namespaces.keys()) if namespaces else 'None'}")
            
            if total_vector_count > 0:
                print("✅ Index contains vectors")
                return True
            else:
                print("❌ Index is empty")
                return False
                
        except Exception as e:
            print(f"✗ Error checking index content: {e}")
            return False

    def analyze_chunk_structure(self, chunks: List[Dict[str, Any]]) -> None:
        """Analyze the structure of chunks to understand metadata fields."""
        if not chunks:
            print("❌ No chunks to analyze")
            return
        
        print(f"🔍 Analyzing {len(chunks)} chunks for metadata structure...")
        
        # Analyze first few chunks
        sample_chunks = chunks[:3]
        
        for i, chunk in enumerate(sample_chunks):
            print(f"\n📋 Chunk {i+1} Analysis:")
            print(f"   Type: {type(chunk)}")
            
            if isinstance(chunk, dict):
                print(f"   Keys: {list(chunk.keys())}")
                
                # Check for None values
                none_fields = [k for k, v in chunk.items() if v is None]
                if none_fields:
                    print(f"   ⚠️  None fields: {none_fields}")
                
                # Check for empty strings
                empty_fields = [k for k, v in chunk.items() if v == ""]
                if empty_fields:
                    print(f"   ⚠️  Empty fields: {empty_fields}")
                
                # Show sample values
                print(f"   Sample values:")
                for key in list(chunk.keys())[:5]:  # First 5 keys
                    value = chunk[key]
                    value_type = type(value).__name__
                    value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"     {key}: {value_type} = {value_preview}")
            else:
                print(f"   Attributes: {dir(chunk)}")
                # Try to get common attributes
                for attr in ['file_name', 'text', 'chunk_summary', 'keywords']:
                    if hasattr(chunk, attr):
                        value = getattr(chunk, attr)
                        print(f"     {attr}: {type(value).__name__} = {str(value)[:50]}...")
        
        print(f"\n📊 Summary:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Sample analyzed: {len(sample_chunks)}")
        print(f"   Chunk type: {type(chunks[0])}")
    
    def get_chunks_from_namespace(self, namespace: str = None) -> List[Dict[str, Any]]:
        """Get chunks from a specific namespace."""
        if not self.index:
            print("❌ Index not initialized")
            return []
        
        try:
            # Use default namespace if none specified
            if not namespace:
                namespace = "__default__"
            
            print(f"🔍 Fetching chunks from namespace: {namespace}")
            
            # Get index stats to check namespace
            stats = self.index.describe_index_stats()
            namespaces = stats.namespaces
            
            if namespace not in namespaces:
                print(f"⚠️ Namespace '{namespace}' not found. Available namespaces: {list(namespaces.keys()) if namespaces else 'None'}")
                return []
            
            # For now, return a sample of chunks (in production, you'd implement proper fetching)
            # This is a placeholder - you'd typically use query or fetch methods
            print(f"✅ Found namespace '{namespace}' with {namespaces[namespace].vector_count} vectors")
            
            # Return sample data for testing
            sample_chunks = [
                {
                    'id': f'sample_chunk_1',
                    'text': 'Sample financial data from Q1 2025 report showing revenue growth and operational improvements.',
                    'metadata': {
                        'section_type': 'Summary',
                        'file_name': 'sample_document.pdf',
                        'chunk_index': 1
                    },
                    'score': 1.0
                },
                {
                    'id': f'sample_chunk_2',
                    'text': 'Financial tables display revenue data with period-over-period comparisons and segment breakdowns.',
                    'metadata': {
                        'section_type': 'Table',
                        'file_name': 'sample_document.pdf',
                        'chunk_index': 2,
                        'table_id': 'revenue_table_1'
                    },
                    'score': 1.0
                }
            ]
            
            print(f"✅ Returning {len(sample_chunks)} sample chunks from namespace '{namespace}'")
            return sample_chunks
            
        except Exception as e:
            print(f"✗ Error getting chunks from namespace: {e}")
            return []