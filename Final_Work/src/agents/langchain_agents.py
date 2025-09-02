#!/usr/bin/env python3
"""
LangChain-Enhanced Agents for the Hybrid RAG System
Upgrades existing agents to use LangChain while maintaining current architecture
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Import existing agent functions
from .summary_agent import run_summary
from .needle_agent import run_needle
from .table_qa_agent import run_table_qa

logger = logging.getLogger(__name__)

class LangChainEnhancedAgents:
    """Enhanced agents using LangChain while maintaining current architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.environ.get('OPENAI_API_KEY')
        )
        
        # Store contexts for tools to access
        self.current_contexts = []
        self.current_query = ""
        self.current_intent = ""
        
        # Initialize tools for each agent type
        self.tools = self._create_tools()
        
        # Initialize the main agent
        self.agent = self._initialize_agent()
        
        # Store retriever for hybrid search
        self.retriever = None
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for each agent type following Middle Course pattern."""
        tools = [
            # Summary Agent Tools
            Tool(
                name="Summary_Generator",
                func=self._enhanced_summary,
                description="Generate comprehensive summaries from retrieved content. Use this for overview, executive summary, or general information requests. Call with: 'summary'"
            ),
            
            # Needle Agent Tools
            Tool(
                name="Information_Extractor",
                func=self._enhanced_needle,
                description="Extract specific information, find precise answers, or locate specific paragraphs/anchors. Use this for exact facts, numbers, dates, or specific details. Call with: 'extract'"
            ),
            
            # Table QA Agent Tools
            Tool(
                name="Table_Analyzer",
                func=self._enhanced_table_qa,
                description="Analyze tables, perform quantitative analysis, or answer questions about numerical data. Use this for calculations, statistics, or table-related queries. Call with: 'analyze'"
            )
        ]
        return tools
    
    def _initialize_agent(self):
        """Initialize the LangChain agent."""
        try:
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent_type="openai-functions",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,  # Reduced from 10 to prevent loops
                max_execution_time=30,  # Reduced from 60 to prevent timeouts
                early_stopping_method="generate"  # Stop early if good answer generated
            )
            logger.info("✓ LangChain agent initialized successfully")
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agent: {e}")
            return None
    
    def set_retriever(self, retriever):
        """Set the retriever for hybrid search functionality."""
        self.retriever = retriever
        logger.info("✓ Retriever set for hybrid search")
    
    def _enhanced_summary(self, action: str) -> str:
        """Enhanced summary generation using LangChain."""
        try:
            if not self.current_contexts:
                return "No context available for summary generation."
            
            # Use existing summary agent for better results
            result = run_summary(self.current_query, self.current_contexts)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _enhanced_needle(self, action: str) -> str:
        """Enhanced information extraction using LangChain."""
        try:
            if not self.current_contexts:
                return "No context available for information extraction."
            
            # Use existing needle agent for better results
            result = run_needle(self.current_query, self.current_contexts)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced needle extraction failed: {e}")
            return f"Information extraction failed: {str(e)}"
    
    def _enhanced_table_qa(self, action: str) -> str:
        """Enhanced table analysis using LangChain."""
        try:
            if not self.current_contexts:
                return "No context available for table analysis."
            
            # Use existing table QA agent for better results
            result = run_table_qa(self.current_query, self.current_contexts)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced table QA failed: {e}")
            return f"Table analysis failed: {str(e)}"
    
    def process_with_langchain(self, query: str, intent: str, contexts: List[Dict] = None) -> str:
        """Process query using LangChain agent."""
        try:
            if not self.agent:
                logger.warning("LangChain agent not available, using fallback")
                return self._fallback_processing(query, intent, contexts)
            
            # Store current query and contexts for tools to access
            self.current_query = query
            self.current_intent = intent
            self.current_contexts = contexts or []
            
            # Create a focused prompt that guides the agent to use the right tool
            if intent == 'summary':
                agent_prompt = f"""
                You need to generate a comprehensive summary for the query: "{query}"
                
                Use the Summary_Generator tool to create a detailed summary from the available context.
                """
            elif intent == 'needle':
                agent_prompt = f"""
                You need to extract specific information for the query: "{query}"
                
                Use the Information_Extractor tool to find the exact information requested.
                """
            elif intent == 'table':
                agent_prompt = f"""
                You need to analyze tables and quantitative data for the query: "{query}"
                
                Use the Table_Analyzer tool to perform the analysis.
                """
            else:
                agent_prompt = f"""
                Answer the following query: "{query}"
                
                Based on the intent ({intent}), use the most appropriate tool to provide a comprehensive answer.
                """
            
            # Use the LangChain agent
            result = self.agent.run(agent_prompt)
            return result
            
        except Exception as e:
            logger.error(f"LangChain processing failed: {e}")
            return self._fallback_processing(query, intent, contexts)
    
    def _fallback_processing(self, query: str, intent: str, contexts: List[Dict] = None) -> str:
        """Fallback to existing agent functions."""
        logger.info(f"Using fallback processing for intent: {intent}")
        
        if intent == 'summary':
            return run_summary(query, contexts or [])
        elif intent == 'needle':
            return run_needle(query, contexts or [])
        else:  # table
            return run_table_qa(query, contexts or [])
