"""
LangChain Tool Definition for PPD Prediction Agent

This module provides a LangChain tool wrapper for the PPD agent,
making it easy to integrate with LangChain agents and chains.
"""

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from ppd_agent import PPDAgent


class PPDPredictionInput(BaseModel):
    """Input schema for PPD prediction tool."""
    age: str = Field(description="Age group (e.g., '25-30', '30-35', etc.)")
    feeling_sad: str = Field(description="Feeling sad or tearful: Yes/No/Sometimes")
    irritable: str = Field(description="Irritable towards baby & partner: Yes/No/Sometimes")
    trouble_sleeping: str = Field(description="Trouble sleeping: Yes/No/Two or more days a week")
    concentration: str = Field(description="Problems concentrating: Yes/No/Often")
    appetite: str = Field(description="Overeating or loss of appetite: Yes/No/Not at all")
    feeling_anxious: str = Field(description="Feeling anxious: Yes/No")
    guilt: str = Field(description="Feeling of guilt: Yes/No/Maybe")
    bonding: str = Field(description="Problems of bonding with baby: Yes/No/Sometimes")
    suicide_attempt: str = Field(description="Suicide attempt: Yes/No/Not interested to say")


class PPDPredictionTool(BaseTool):
    """
    LangChain tool for PPD risk prediction.
    
    This tool can be used with LangChain agents to predict
    postpartum depression risk based on patient symptoms.
    """
    
    name: str = "predict_ppd_risk"
    description: str = (
        "Predicts postpartum depression (PPD) risk based on patient symptoms and demographics. "
        "Returns risk score (0-100%), risk level (Low/Moderate/High/Very High), "
        "feature importance, and personalized explanation. "
        "Use this tool when you need to assess PPD risk for a patient."
    )
    args_schema: Type[BaseModel] = PPDPredictionInput
    
    agent: Optional[PPDAgent] = None
    
    def __init__(self, ppd_agent: PPDAgent, **kwargs):
        """
        Initialize the tool with a PPD agent.
        
        Args:
            ppd_agent: PPDAgent instance
            **kwargs: Additional arguments for BaseTool
        """
        super().__init__(**kwargs)
        self.agent = ppd_agent
    
    def _run(self, 
             age: str,
             feeling_sad: str,
             irritable: str,
             trouble_sleeping: str,
             concentration: str,
             appetite: str,
             feeling_anxious: str,
             guilt: str,
             bonding: str,
             suicide_attempt: str) -> str:
        """
        Execute the tool.
        
        Args:
            All input parameters for PPD prediction
        
        Returns:
            Formatted string with prediction results
        """
        if self.agent is None:
            return "Error: PPD Agent not initialized"
        
        try:
            result = self.agent.predict(
                age=age,
                feeling_sad=feeling_sad,
                irritable=irritable,
                trouble_sleeping=trouble_sleeping,
                concentration=concentration,
                appetite=appetite,
                feeling_anxious=feeling_anxious,
                guilt=guilt,
                bonding=bonding,
                suicide_attempt=suicide_attempt
            )
            
            # Format result as a readable string
            output = f"""
PPD Risk Assessment:
- Risk Score: {result['risk_percentage']}%
- Risk Level: {result['risk_level']}
- Prediction: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}

Explanation:
{result['explanation']}

Top Contributing Factors:
"""
            for i, feature in enumerate(result['feature_importance'][:5], 1):
                feature_name = feature['feature'].split('_')[-1] if '_' in feature['feature'] else feature['feature']
                output += f"{i}. {feature_name}: {feature['impact']} risk (contribution: {abs(feature['shap_value']):.4f})\n"
            
            return output.strip()
            
        except Exception as e:
            return f"Error during prediction: {str(e)}"
    
    async def _arun(self, 
                    age: str,
                    feeling_sad: str,
                    irritable: str,
                    trouble_sleeping: str,
                    concentration: str,
                    appetite: str,
                    feeling_anxious: str,
                    guilt: str,
                    bonding: str,
                    suicide_attempt: str) -> str:
        """Async version of _run."""
        return self._run(
            age=age,
            feeling_sad=feeling_sad,
            irritable=irritable,
            trouble_sleeping=trouble_sleeping,
            concentration=concentration,
            appetite=appetite,
            feeling_anxious=feeling_anxious,
            guilt=guilt,
            bonding=bonding,
            suicide_attempt=suicide_attempt
        )


def create_langchain_tool(ppd_agent: PPDAgent) -> PPDPredictionTool:
    """
    Create a LangChain tool from a PPD agent.
    
    Args:
        ppd_agent: PPDAgent instance
    
    Returns:
        PPDPredictionTool instance
    """
    return PPDPredictionTool(ppd_agent=ppd_agent)


# Example usage with LangChain agent:
"""
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Create tool
tool = create_langchain_tool(ppd_agent)

# Initialize agent with the tool
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.run(
    "What is the PPD risk for a 30-year-old patient who feels sad, "
    "is irritable, has trouble sleeping, and reports feeling anxious?"
)
"""

