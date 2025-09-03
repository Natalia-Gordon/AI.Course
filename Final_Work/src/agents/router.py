from typing import Literal, Optional
import logging
import time
from enum import Enum
from utils.agent_logger import log_router_agent

# Try to import LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    """Intent types for query classification."""
    SUMMARY = "summary"
    TABLE = "table"
    NEEDLE = "needle"

class QueryIntent(BaseModel):
    """Structured output for query intent classification."""
    intent: IntentType = Field(
        description="The primary intent of the query. Choose 'summary' for general overview requests, 'table' for quantitative/statistical analysis, or 'needle' for specific information extraction."
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0 for the intent classification",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )

def route_intent_llm(query: str) -> Literal["summary", "table", "needle"]:
    """
    Route queries using LangChain + OpenAI for intelligent intent classification.
    This is much more accurate than keyword matching.
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available, falling back to keyword-based routing")
        return route_intent_keywords(query)
    
    try:
        # Initialize OpenAI model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=150
        )
        
        # Create the classification prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query classifier for a financial document analysis system. 
Your task is to classify user queries into one of three categories:

1. **SUMMARY** - Queries asking for general overview, summaries, highlights, or broad descriptions
   Examples: "Summarize the report", "Give me an overview", "What are the key highlights", "Executive summary"

2. **TABLE** - Queries asking for quantitative analysis, calculations, statistics, or table-related information
   Examples: "Calculate the average", "Show me tables", "What are the statistics", "Compare the numbers"

3. **NEEDLE** - Queries asking for specific, precise information, locations, or exact details
   Examples: "What was the revenue in Q1?", "Find the page number for", "Where is the executive summary", "Show me specific data"

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{{
  "intent": "summary|table|needle",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}

Do not include any other text or explanations outside the JSON."""),
            ("user", "Classify this query: {query}")
        ])
        
        # Create the output parser
        output_parser = JsonOutputParser(pydantic_object=QueryIntent)
        
        # Create the chain
        chain = prompt | llm | output_parser
        
        # Get the classification
        result = chain.invoke({"query": query})
        
        # Validate the result
        if isinstance(result, dict) and 'intent' in result:
            intent = result['intent']
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'No reasoning provided')
            
            logger.info(f"LLM Intent Classification: {intent} (confidence: {confidence:.2f})")
            logger.info(f"Reasoning: {reasoning}")
            
            return intent
        else:
            # Try to extract intent from text response using regex
            import re
            if isinstance(result, str):
                # Look for intent patterns in the text
                if re.search(r'\*\*NEEDLE\*\*', result, re.IGNORECASE):
                    logger.info("Extracted NEEDLE intent from text response")
                    return "needle"
                elif re.search(r'\*\*SUMMARY\*\*', result, re.IGNORECASE):
                    logger.info("Extracted SUMMARY intent from text response")
                    return "summary"
                elif re.search(r'\*\*TABLE\*\*', result, re.IGNORECASE):
                    logger.info("Extracted TABLE intent from text response")
                    return "table"
            
            logger.warning(f"Invalid LLM response format: {result}")
            raise ValueError("Invalid response format")
        
    except Exception as e:
        logger.warning(f"LLM-based routing failed: {e}, falling back to keyword-based routing")
        return route_intent_keywords(query)

def route_intent_keywords(query: str) -> Literal["summary", "table", "needle"]:
    """
    Fallback keyword-based routing for when LangChain is not available.
    This is the original implementation.
    """
    q = query.lower()
    
    # PRIORITY 1: Location-specific queries (NEEDLE) - check these FIRST
    location_patterns = [
        "page number for", "location of", "where to find", "reference to",
        "find the page", "where can i find", "in which section"
    ]
    if any(pattern in q for pattern in location_patterns):
        return "needle"
    
    # PRIORITY 2: Specific section requests (NEEDLE)
    specific_sections = [
        "executive summary", "financial highlights", "management discussion",
        "risk factors", "notes to financial statements"
    ]
    if any(section in q for section in specific_sections):
        return "needle"
    
    # PRIORITY 3: Table queries - look for quantitative analysis (check before summary)
    table_keywords = [
        # English keywords
        "table", "average", "avg", "sum", "total", "median", "percent", 
        "percentage", "chart", "graph", "calculate", "compute", "statistics",
        "compare", "trend", "growth", "decline", "data", "numbers", "figures",
        "show me", "display", "present", "revenue data", "financial data",
        "revenue", "profit", "income", "financial", "metrics", "values",
        # Hebrew keywords
        "טבלה", "נתונים", "מספרים", "סטטיסטיקה", "חישוב", "ממוצע", 
        "סכום", "סה״כ", "אחוז", "השוואה", "מגמה", "צמיחה", "ירידה",
        "הצג לי", "מה הנתונים", "נתוני", "מספרי", "סטטיסטיקות"
    ]
    if any(keyword in q for keyword in table_keywords):
        return "table"
    
    # PRIORITY 4: Summary queries - look for summary-related keywords
    summary_keywords = [
        "summarize", "overview", "highlight", "brief", "general", 
        "describe", "key points", "main points", "tell me about"
    ]
    if any(keyword in q for keyword in summary_keywords):
        return "summary"
    
    # PRIORITY 5: Needle queries - look for specific information requests
    needle_keywords = [
        "page ", "tableid", "figure", "anchor", "paragraph", "section",
        "find", "locate", "where is", "specific", "exact", "precise",
        "what was", "how much", "what is", "find information",
        "page number", "which page", "on what page", "in which section"
    ]
    if any(keyword in q for keyword in needle_keywords):
        return "needle"
    
    # Default to summary for general queries
    return "summary"

@log_router_agent
def route_intent(query: str) -> Literal["summary", "table", "needle"]:
    """
    Main routing function that tries LLM-based classification first,
    then falls back to keyword-based routing if needed.
    """
    try:
        # Try LLM-based routing first
        return route_intent_llm(query)
    except Exception as e:
        logger.warning(f"LLM routing failed: {e}, using keyword fallback")
        return route_intent_keywords(query)

# Backward compatibility
def route_intent_old(query: str) -> Literal["summary", "table", "needle"]:
    """Legacy function for backward compatibility."""
    return route_intent_keywords(query)
