#!/usr/bin/env python3
"""
Base Agent Class for Hybrid RAG System
Provides consistent structure, logging, and error handling for all agents
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from utils.logger import get_logger, log_agent_action, log_performance


class BaseAgent(ABC):
    """Base class for all RAG agents with consistent logging and error handling"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = get_logger(name)
        
        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        self.logger.info(f"🚀 {self.name.replace('_', ' ').title()} Agent initialized")
    
    @abstractmethod
    def process_query(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Process a query with given contexts - must be implemented by subclasses"""
        pass
    
    def execute(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Execute the agent with logging and performance tracking"""
        start_time = time.time()
        self.execution_count += 1
        
        # Log execution start
        log_agent_action(self.name, f"Starting query processing", 
                        query=query, contexts_count=len(contexts))
        
        try:
            # Process the query
            result = self.process_query(query, contexts)
            
            # Track performance
            duration = time.time() - start_time
            self.total_execution_time += duration
            
            # Log success
            log_performance(self.name, "query_processing", duration,
                          query_length=len(query),
                          result_length=len(result),
                          contexts_used=len(contexts))
            
            self.logger.info(f"✅ Query processed successfully in {duration:.3f}s")
            return result
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self.logger.error(f"❌ Query processing failed after {duration:.3f}s: {str(e)}")
            self.logger.debug(f"Error details: {type(e).__name__}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0.0)
        
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_time
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.logger.info(f"📊 Statistics reset for {self.name}")


class AgentRegistry:
    """Registry for managing all agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = get_logger('agent_registry')
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent in the registry"""
        self.agents[agent.name] = agent
        self.logger.info(f"✅ Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self.agents.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents"""
        return {name: agent.get_stats() for name, agent in self.agents.items()}
