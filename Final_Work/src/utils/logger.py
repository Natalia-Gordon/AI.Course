#!/usr/bin/env python3
"""
Comprehensive Logging System for Hybrid RAG
Provides date-based, component-specific logging with rotation and configuration
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


class HybridRAGLogger:
    """Comprehensive logging system for Hybrid RAG components"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = self._load_config(config_path)
        self.logging_config = self.config.get('logging', {})
        self.base_dir = self.logging_config.get('base_dir', 'logs')
        self.date_format = self.logging_config.get('date_format', '%Y-%m-%d')
        self.file_pattern = self.logging_config.get('file_pattern', '{component}_{date}.log')
        
        # Ensure logs directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize loggers cache
        self._loggers = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {}
    
    def get_logger(self, component: str, level: Optional[str] = None) -> logging.Logger:
        """Get or create a logger for a specific component"""
        if component in self._loggers:
            return self._loggers[component]
        
        # Create new logger
        logger = logging.getLogger(component)
        
        # Set log level
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        else:
            # Get level from config or default to INFO
            config_level = self.logging_config.get('levels', {}).get(component, 'INFO')
            logger.setLevel(getattr(logging, config_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Add console handler if enabled
        if self.logging_config.get('console', {}).get('enabled', True):
            console_handler = self._create_console_handler(component)
            logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if self.logging_config.get('file', {}).get('enabled', True):
            file_handler = self._create_file_handler(component)
            logger.addHandler(file_handler)
        
        # Store logger in cache
        self._loggers[component] = logger
        
        return logger
    
    def _create_console_handler(self, component: str) -> logging.StreamHandler:
        """Create console handler with appropriate formatting"""
        console_handler = logging.StreamHandler()
        
        # Set console log level
        console_level = self.logging_config.get('console', {}).get('level', 'INFO')
        console_handler.setLevel(getattr(logging, console_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            self.logging_config.get('format', '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'),
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        return console_handler
    
    def _create_file_handler(self, component: str) -> logging.handlers.RotatingFileHandler:
        """Create rotating file handler with date-based naming"""
        # Check if this component should use unified evaluation logging
        evaluation_logging = self.logging_config.get('evaluation_logging', {})
        if evaluation_logging.get('unified_file', False) and component in evaluation_logging.get('components', []):
            # Use unified evaluation log file
            current_date = datetime.now().strftime(self.date_format)
            filename = evaluation_logging['filename'].format(date=current_date)
            filepath = os.path.join(self.base_dir, filename)
        else:
            # Use component-specific log file
            current_date = datetime.now().strftime(self.date_format)
            filename = self.file_pattern.format(component=component, date=current_date)
            filepath = os.path.join(self.base_dir, filename)
        
        # Create rotating file handler
        max_bytes = self.logging_config.get('rotation', {}).get('max_size_mb', 10) * 1024 * 1024
        backup_count = self.logging_config.get('rotation', {}).get('backup_count', 7)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Set file log level
        file_level = self.logging_config.get('file', {}).get('level', 'INFO')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        # Create formatter for file (include date)
        formatter = logging.Formatter(
            self.logging_config.get('format', '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        return file_handler
    
    def log_system_start(self, component: str, **kwargs):
        """Log system startup information"""
        logger = self.get_logger(component)
        logger.info(f"🚀 {component.replace('_', ' ').title()} System Starting")
        
        # Log additional startup parameters
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")
    
    def log_system_stop(self, component: str, **kwargs):
        """Log system shutdown information"""
        logger = self.get_logger(component)
        logger.info(f"🛑 {component.replace('_', ' ').title()} System Stopping")
        
        # Log additional shutdown parameters
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")
    
    def log_agent_action(self, component: str, action: str, **kwargs):
        """Log agent actions with parameters"""
        logger = self.get_logger(component)
        logger.info(f"🤖 {action}")
        
        # Log action parameters
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")
    
    def log_evaluation_result(self, component: str, metric: str, value: float, target: float, status: str):
        """Log evaluation results"""
        logger = self.get_logger(component)
        status_icon = "✅" if status == "PASS" else "❌"
        logger.info(f"{status_icon} {metric}: {value:.3f} (Target: ≥{target:.2f})")
    
    def log_error(self, component: str, error: Exception, context: str = ""):
        """Log errors with context"""
        logger = self.get_logger(component)
        logger.error(f"❌ Error in {context}: {str(error)}")
        logger.debug(f"Error details: {type(error).__name__}", exc_info=True)
    
    def log_performance(self, component: str, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        logger = self.get_logger(component)
        logger.info(f"⏱️  {operation} completed in {duration:.3f}s")
        
        # Log additional performance metrics
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")


# Global logger instance
_global_logger = None

def get_logger(component: str, level: Optional[str] = None) -> logging.Logger:
    """Get logger for a specific component (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    return _global_logger.get_logger(component, level)

def log_system_start(component: str, **kwargs):
    """Log system startup (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_system_start(component, **kwargs)

def log_system_stop(component: str, **kwargs):
    """Log system shutdown (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_system_stop(component, **kwargs)

def log_agent_action(component: str, action: str, **kwargs):
    """Log agent actions (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_agent_action(component, action, **kwargs)

def log_evaluation_result(component: str, metric: str, value: float, target: float, status: str):
    """Log evaluation results (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_evaluation_result(component, metric, value, target, status)

def log_error(component: str, error: Exception, context: str = ""):
    """Log errors (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_error(component, error, context)

def log_performance(component: str, operation: str, duration: float, **kwargs):
    """Log performance metrics (global access)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HybridRAGLogger()
    _global_logger.log_performance(component, operation, duration, **kwargs)
