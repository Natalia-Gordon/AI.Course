#!/usr/bin/env python3
"""
Agent Logging Wrapper
Provides logging functionality for agent functions without changing their structure
"""

import time
import functools
from typing import Callable, Any
from utils.logger import get_logger, log_agent_action, log_performance


def log_agent_execution(agent_name: str):
    """Decorator to add logging to agent functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(agent_name)
            
            # Log agent action start
            query = args[0] if args else "No query"
            log_agent_action(agent_name, f"Starting {func.__name__}", query=query)
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log success and performance
                duration = time.time() - start_time
                log_performance(agent_name, func.__name__, duration, 
                              query_length=len(str(query)),
                              result_length=len(str(result)))
                
                logger.info(f"✅ {func.__name__} completed successfully in {duration:.3f}s")
                return result
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                logger.error(f"❌ {func.__name__} failed after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Pre-configured decorators for each agent
def log_summary_agent(func: Callable) -> Callable:
    return log_agent_execution('summary_agent')(func)

def log_needle_agent(func: Callable) -> Callable:
    return log_agent_execution('needle_agent')(func)

def log_table_qa_agent(func: Callable) -> Callable:
    return log_agent_execution('table_qa_agent')(func)

def log_router_agent(func: Callable) -> Callable:
    return log_agent_execution('router_agent')(func)
