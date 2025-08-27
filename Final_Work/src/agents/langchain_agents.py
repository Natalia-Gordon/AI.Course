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
                description="Generate comprehensive summaries from retrieved content. Use this for overview, executive summary, or general information requests."
            ),
            
            # Needle Agent Tools
            Tool(
                name="Information_Extractor",
                func=self._enhanced_needle,
                description="Extract specific information, find precise answers, or locate specific paragraphs/anchors. Use this for exact facts, numbers, dates, or specific details."
            ),
            
            # Table QA Agent Tools
            Tool(
                name="Table_Analyzer",
                func=self._enhanced_table_qa,
                description="Analyze tables, perform quantitative analysis, or answer questions about numerical data. Use this for calculations, statistics, or table-related queries."
            ),
            
            # Hybrid Search Tool
            Tool(
                name="Hybrid_Search",
                func=self._hybrid_search,
                description="Perform hybrid dense+sparse search to find relevant document chunks. Use this to retrieve context before processing."
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
                max_iterations=10,
                max_execution_time=60
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
    
    def _hybrid_search(self, query: str) -> str:
        """Enhanced hybrid search tool for retrieving relevant chunks."""
        try:
            if not self.retriever:
                return "Hybrid search not available. Please set retriever first."
            
            # Perform hybrid search using the existing retriever
            hits = self.retriever.search(
                query,
                k_dense=self.config['retrieval']['dense_k'],
                k_sparse=self.config['retrieval']['sparse_k'],
                final_k=self.config['retrieval']['final_k']
            )
            
            # Format results for display
            result = f"Found {len(hits)} relevant chunks for query: '{query}'\n\n"
            
            for i, hit in enumerate(hits[:5]):  # Show top 5 results
                summary = hit.get('chunk_summary', hit.get('text', '')[:150])
                source = f"Source: {hit.get('file_name', 'Unknown')}"
                page = f"Page: {hit.get('page_number', 'Unknown')}"
                section = f"Type: {hit.get('section_type', 'Unknown')}"
                
                result += f"{i+1}. {summary}...\n"
                result += f"   {source} | {page} | {section}\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return f"Hybrid search failed: {str(e)}"
    
    def _enhanced_summary(self, query: str, contexts: str = None) -> str:
        """Enhanced summary generation using LangChain."""
        try:
            if not contexts:
                return "Please provide contexts for summary generation."
            
            # Parse contexts if provided as string
            if isinstance(contexts, str):
                context_text = contexts
            else:
                context_text = str(contexts)
            
            # Use LangChain's summarize chain
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Split context into manageable chunks
            docs = text_splitter.create_documents([context_text])
            
            # Use map-reduce for better summaries
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=self._get_summary_map_prompt(),
                combine_prompt=self._get_summary_combine_prompt(),
                verbose=False
            )
            
            result = chain.invoke({"input_documents": docs})
            return result["output_text"]
            
        except Exception as e:
            logger.error(f"Enhanced summary generation failed: {e}")
            # Fallback to existing summary agent
            return f"Enhanced summary failed, using fallback: {str(e)}"
    
    def _enhanced_needle(self, query: str, contexts: str = None) -> str:
        """Enhanced information extraction using LangChain."""
        try:
            if not contexts:
                return "Please provide contexts for information extraction."
            
            # Create a custom prompt for precise information extraction
            prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are an expert information extractor. Given the following query and context, extract the most precise and relevant information.
                
                Query: {query}
                Context: {context}
                
                Instructions:
                1. Find the exact information requested
                2. Include specific numbers, dates, names, and facts
                3. Be precise and accurate
                4. If the information is not in the context, say so clearly
                
                Answer:"""
            )
            
            # Format the prompt
            formatted_prompt = prompt.format(
                query=query,
                context=contexts[:3000]  # Limit context length
            )
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Enhanced needle extraction failed: {e}")
            # Fallback to existing needle agent
            return f"Enhanced extraction failed, using fallback: {str(e)}"
    
    def _enhanced_table_qa(self, query: str, contexts: str = None) -> str:
        """Enhanced table analysis using LangChain."""
        try:
            if not contexts:
                return "Please provide contexts for table analysis."
            
            # Create a custom prompt for table analysis
            prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are an expert financial analyst. Given the following query and context (which may contain tables), provide a quantitative analysis.
                
                Query: {query}
                Context: {context}
                
                Instructions:
                1. If tables are present, analyze the numerical data
                2. Perform calculations if requested
                3. Provide specific numbers and percentages
                4. Reference table IDs and page numbers when available
                5. Be precise with financial figures
                
                Analysis:"""
            )
            
            # Format the prompt
            formatted_prompt = prompt.format(
                query=query,
                context=contexts[:3000]  # Limit context length
            )
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Enhanced table QA failed: {e}")
            # Fallback to existing needle agent
            return f"Enhanced table analysis failed, using fallback: {str(e)}"
    
    def _get_summary_map_prompt(self) -> PromptTemplate:
        """Get the map prompt for summary generation."""
        return PromptTemplate(
            input_variables=["text"],
            template="""
            Summarize the following text in a concise way:
            
            {text}
            
            Summary:"""
        )
    
    def _get_summary_combine_prompt(self) -> PromptTemplate:
        """Get the combine prompt for summary generation."""
        return PromptTemplate(
            input_variables=["text"],
            template="""
            Create a comprehensive summary from the following summaries:
            
            {text}
            
            Comprehensive Summary:"""
        )
    
    def process_with_langchain(self, query: str, intent: str, contexts: List[Dict] = None) -> str:
        """Process query using LangChain agent."""
        try:
            if not self.agent:
                logger.warning("LangChain agent not available, using fallback")
                return self._fallback_processing(query, intent, contexts)
            
            # Prepare context for the agent
            context_text = self._prepare_context(contexts) if contexts else "No context provided"
            
            # Create a prompt that guides the agent to use the right tool
            agent_prompt = f"""
            Based on the query and context, use the appropriate tool to provide a comprehensive answer.
            
            Query: {query}
            Intent: {intent}
            Context: {context_text}
            
            Please analyze this and provide a detailed response using the most appropriate tool.
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
    
    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context for LangChain processing."""
        if not contexts:
            return "No context available"
        
        context_parts = []
        for i, context in enumerate(contexts[:5]):  # Limit to 5 contexts
            text = context.get('text', context.get('chunk_summary', ''))
            source = f"Source: {context.get('file_name', 'Unknown')} | Page: {context.get('page_number', 'Unknown')}"
            context_parts.append(f"Context {i+1}:\n{text[:500]}...\n{source}\n")
        
        return "\n".join(context_parts)
