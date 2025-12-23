"""
FastAPI REST API Server for PPD Prediction Agent

This provides a REST API interface for the PPD prediction agent,
making it accessible via HTTP requests.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import os
from ppd_agent import PPDAgent

# Global agent instance (will be initialized on startup)
agent: Optional[PPDAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI (replaces deprecated on_event).
    Handles startup and shutdown events.
    """
    # Startup
    global agent
    if agent is None:
        # Try to load from saved file
        agent_file = "output/agents/ppd_agent.pkl"
        if os.path.exists(agent_file):
            try:
                print(f"Loading agent from {agent_file}...")
                agent = PPDAgent.load(agent_file)
                print("Agent loaded successfully!")
            except Exception as e:
                print(f"WARNING: Could not load agent from {agent_file}: {e}")
                print("WARNING: Please initialize the agent using initialize_agent() or train a new model.")
        else:
            print("WARNING: Agent not initialized and no saved agent file found.")
            print("WARNING: Please initialize the agent using initialize_agent() or train a new model.")
            print("WARNING: The API will start but prediction endpoints will return errors.")
    
    yield
    
    # Shutdown (if needed)
    # Cleanup code can go here


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="PPD Prediction API",
    description="API for Postpartum Depression Risk Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    age: str = Field(..., description="Age group (e.g., '25-30', '30-35')")
    feeling_sad: str = Field(..., description="Feeling sad or tearful: Yes/No/Sometimes")
    irritable: str = Field(..., description="Irritable towards baby & partner: Yes/No/Sometimes")
    trouble_sleeping: str = Field(..., description="Trouble sleeping: Yes/No/Two or more days a week")
    concentration: str = Field(..., description="Problems concentrating: Yes/No/Often")
    appetite: str = Field(..., description="Overeating or loss of appetite: Yes/No/Not at all")
    feeling_anxious: str = Field(..., description="Feeling anxious: Yes/No")
    guilt: str = Field(..., description="Feeling of guilt: Yes/No/Maybe")
    bonding: str = Field(..., description="Problems of bonding with baby: Yes/No/Sometimes")
    suicide_attempt: str = Field(..., description="Suicide attempt: Yes/No/Not interested to say")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    patients: List[PredictionRequest] = Field(..., description="List of patient data")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    risk_score: float = Field(..., description="PPD risk score (0-1)")
    risk_percentage: float = Field(..., description="PPD risk as percentage")
    risk_level: str = Field(..., description="Risk level: Low/Moderate/High/Very High")
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    feature_importance: List[dict] = Field(..., description="Top 5 feature contributions")
    explanation: str = Field(..., description="Personalized explanation")
    probabilities: dict = Field(..., description="Class probabilities")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[PredictionResponse] = Field(..., description="List of prediction results")




def initialize_agent(ppd_agent: PPDAgent):
    """
    Initialize the global agent instance.
    
    Args:
        ppd_agent: PPDAgent instance
    """
    global agent
    agent = ppd_agent
    print("PPD Agent initialized for API server")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PPD Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/schema": "GET - Get tool schema"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict PPD risk for a single patient.
    
    Args:
        request: Prediction request with patient data
    
    Returns:
        Prediction response with risk score and explanation
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.predict(
            age=request.age,
            feeling_sad=request.feeling_sad,
            irritable=request.irritable,
            trouble_sleeping=request.trouble_sleeping,
            concentration=request.concentration,
            appetite=request.appetite,
            feeling_anxious=request.feeling_anxious,
            guilt=request.guilt,
            bonding=request.bonding,
            suicide_attempt=request.suicide_attempt
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict PPD risk for multiple patients.
    
    Args:
        request: Batch prediction request with list of patients
    
    Returns:
        Batch prediction response with results for all patients
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        input_list = [
            {
                "Age": p.age,
                "Feeling sad or Tearful": p.feeling_sad,
                "Irritable towards baby & partner": p.irritable,
                "Trouble sleeping at night": p.trouble_sleeping,
                "Problems concentrating or making decision": p.concentration,
                "Overeating or loss of appetite": p.appetite,
                "Feeling anxious": p.feeling_anxious,
                "Feeling of guilt": p.guilt,
                "Problems of bonding with baby": p.bonding,
                "Suicide attempt": p.suicide_attempt
            }
            for p in request.patients
        ]
        
        results = agent.batch_predict(input_list)
        return BatchPredictionResponse(
            results=[PredictionResponse(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/schema")
async def get_schema():
    """
    Get OpenAI Function Calling schema for this agent.
    
    Returns:
        Tool schema in OpenAI format
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    return agent.get_tool_schema()


if __name__ == "__main__":
    # This will be called when running the server directly
    # The agent will be automatically loaded from output/agents/ppd_agent.pkl if it exists
    # Otherwise, you can initialize it programmatically:
    #   from api_server import initialize_agent
    #   from ppd_agent import PPDAgent
    #   agent = PPDAgent.load("output/agents/ppd_agent.pkl")
    #   initialize_agent(agent)
    print("Starting PPD Prediction API Server...")
    print("The agent will be automatically loaded from output/agents/ppd_agent.pkl if available.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

