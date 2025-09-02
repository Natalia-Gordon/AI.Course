#!/usr/bin/env python3
"""
Configuration Manager for Hybrid RAG System
Provides type-safe, validated configuration access
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from utils.logger import get_logger


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    provider: str
    model: str
    dimension: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        return cls(
            provider=data.get('provider', 'openai'),
            model=data.get('model', 'text-embedding-3-small'),
            dimension=data.get('dim', 1536)
        )


@dataclass
class PineconeConfig:
    """Pinecone configuration"""
    index_name: str
    region: Optional[str] = None
    cloud: Optional[str] = None
    metric: str = 'cosine'
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PineconeConfig':
        return cls(
            index_name=data.get('index_name', 'financial-reports'),
            region=data.get('region'),
            cloud=data.get('cloud'),
            metric=data.get('metric', 'cosine')
        )


@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    max_chunk_tokens: int
    overlap_tokens: int
    budget_ratio: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkingConfig':
        return cls(
            max_chunk_tokens=data.get('max_chunk_tokens', 400),
            overlap_tokens=data.get('overlap_tokens', 60),
            budget_ratio=data.get('budget_ratio', 0.05)
        )


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    dense_k: int
    sparse_k: int
    final_k: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalConfig':
        return cls(
            dense_k=data.get('dense_k', 10),
            sparse_k=data.get('sparse_k', 10),
            final_k=data.get('final_k', 6)
        )


@dataclass
class LoggingConfig:
    """Logging configuration"""
    base_dir: str
    levels: Dict[str, str]
    file_pattern: str
    date_format: str
    max_size_mb: int
    backup_count: int
    format: str
    console_enabled: bool
    file_enabled: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        levels = data.get('levels', {})
        rotation = data.get('rotation', {})
        console = data.get('console', {})
        file = data.get('file', {})
        
        return cls(
            base_dir=data.get('base_dir', 'logs'),
            levels=levels,
            file_pattern=data.get('file_pattern', '{component}_{date}.log'),
            date_format=data.get('date_format', '%Y-%m-%d'),
            max_size_mb=rotation.get('max_size_mb', 10),
            backup_count=rotation.get('backup_count', 7),
            format=data.get('format', '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'),
            console_enabled=console.get('enabled', True),
            file_enabled=file.get('enabled', True)
        )


class ConfigManager:
    """Manages configuration loading, validation, and access"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger('config_manager')
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Dict[str, Any]] = None
        
        # Load configuration
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in the project structure"""
        possible_paths = [
            'src/config.yaml',
            'config.yaml',
            '../src/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found. Please create src/config.yaml")
    
    def _load_config(self):
        """Load and validate configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['project_name', 'embedding', 'pinecone', 'chunking', 'retrieval']
            missing_sections = [section for section in required_sections if section not in raw_config]
            
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {missing_sections}")
            
            # Parse configuration sections
            self._config = {
                'project_name': raw_config['project_name'],
                'embedding': EmbeddingConfig.from_dict(raw_config['embedding']),
                'pinecone': PineconeConfig.from_dict(raw_config['pinecone']),
                'chunking': ChunkingConfig.from_dict(raw_config['chunking']),
                'retrieval': RetrievalConfig.from_dict(raw_config['retrieval']),
                'logging': LoggingConfig.from_dict(raw_config.get('logging', {})),
                'filters': raw_config.get('filters', {}),
                'financial': raw_config.get('financial', {})
            }
            
            self.logger.info(f"✅ Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            raise
    
    @property
    def project_name(self) -> str:
        """Get project name"""
        return self._config['project_name']
    
    @property
    def embedding(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        return self._config['embedding']
    
    @property
    def pinecone(self) -> PineconeConfig:
        """Get Pinecone configuration"""
        return self._config['pinecone']
    
    @property
    def chunking(self) -> ChunkingConfig:
        """Get chunking configuration"""
        return self._config['chunking']
    
    @property
    def retrieval(self) -> RetrievalConfig:
        """Get retrieval configuration"""
        return self._config['retrieval']
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        return self._config['logging']
    
    @property
    def filters(self) -> Dict[str, Any]:
        """Get filters configuration"""
        return self._config['filters']
    
    @property
    def financial(self) -> Dict[str, Any]:
        """Get financial configuration"""
        return self._config['financial']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self._config.get(key, default)
    
    def validate_environment(self) -> bool:
        """Validate that required environment variables are set"""
        required_vars = ['PINECONE_API_KEY', 'PINECONE_REGION', 'OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        self.logger.info("✅ All required environment variables are set")
        return True
    
    def reload(self):
        """Reload configuration from file"""
        self.logger.info("🔄 Reloading configuration...")
        self._load_config()
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment variables"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key
