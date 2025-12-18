# PPD Prediction Agent Tool

This document describes how to use the Postpartum Depression (PPD) Prediction system as an agent tool.

## Overview

The PPD Agent Tool provides multiple interfaces for integrating the PPD prediction model into your applications:

1. **Standalone Python Agent** - Direct Python API
2. **REST API Server** - HTTP API for web services
3. **LangChain Tool** - Integration with LangChain agents
4. **OpenAI Function Calling** - Schema for OpenAI agents
5. **Gradio Interface** - Web UI (existing)

## Quick Start

### 1. Standalone Python Usage

```python
from ppd_agent import PPDAgent, create_agent_from_training
from MLmodel import create_pipeline, train_and_evaluate
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup (after training)
agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))

# Make prediction
result = agent.predict(
    age="30-35",
    feeling_sad="Yes",
    irritable="Yes",
    trouble_sleeping="Yes",
    concentration="Yes",
    appetite="No",
    feeling_anxious="Yes",
    guilt="Yes",
    bonding="Sometimes",
    suicide_attempt="No"
)

print(f"Risk: {result['risk_percentage']}%")
print(f"Level: {result['risk_level']}")
print(f"Explanation: {result['explanation']}")
```

### 2. REST API Server

```bash
# Start the API server
python api_server.py
```

Then make HTTP requests:

```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    "age": "30-35",
    "feeling_sad": "Yes",
    "irritable": "Yes",
    "trouble_sleeping": "Yes",
    "concentration": "Yes",
    "appetite": "No",
    "feeling_anxious": "Yes",
    "guilt": "Yes",
    "bonding": "Sometimes",
    "suicide_attempt": "No"
})

result = response.json()
print(f"Risk: {result['risk_percentage']}%")
```

**API Endpoints:**
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /schema` - Get OpenAI function schema

### 3. LangChain Integration

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_tool import create_langchain_tool

# Create tool
agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
tool = create_langchain_tool(agent)

# Initialize LangChain agent
llm = OpenAI(temperature=0)
langchain_agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = langchain_agent.run(
    "What is the PPD risk for a 30-year-old patient who feels sad, "
    "is irritable, and has trouble sleeping?"
)
```

### 4. OpenAI Function Calling

```python
import openai
from ppd_agent import create_agent_from_training

# Get schema
agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
schema = agent.get_tool_schema()

# Use with OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Assess PPD risk for a patient..."}],
    functions=[schema],
    function_call={"name": "predict_ppd_risk"}
)
```

## Installation

Add to `requirements.txt`:

```
fastapi
uvicorn
pydantic
langchain
openai
```

Install:

```bash
pip install -r requirements.txt
```

## File Structure

```
Hackathon/
├── ppd_agent.py          # Core agent class
├── api_server.py         # FastAPI REST API
├── langchain_tool.py     # LangChain tool wrapper
├── agent_examples.py     # Usage examples
├── main.py               # Main script (creates agent)
└── AGENT_TOOL_README.md  # This file
```

## Agent Methods

### `predict(...)`
Make a single prediction with individual parameters.

### `predict_from_dict(input_dict)`
Make a prediction from a dictionary of inputs.

### `batch_predict(input_list)`
Make predictions for multiple patients at once.

### `get_tool_schema()`
Get OpenAI Function Calling schema.

### `save(filepath)`
Save the agent to a file.

### `load(filepath)`
Load an agent from a file.

## Response Format

All prediction methods return a dictionary with:

```python
{
    "risk_score": 0.8548,           # 0-1 probability
    "risk_percentage": 85.48,        # Percentage
    "risk_level": "High",            # Low/Moderate/High/Very High
    "prediction": 1,                 # Binary (0 or 1)
    "feature_importance": [          # Top 5 features
        {
            "feature": "Feeling_anxious_Yes",
            "shap_value": 0.1234,
            "impact": "increases",
            "abs_contribution": 0.1234
        },
        ...
    ],
    "explanation": "The model identifies a high risk...",
    "probabilities": {
        "no_depression": 14.52,
        "depression": 85.48
    }
}
```

## Examples

See `agent_examples.py` for complete examples of all usage patterns.

Run examples:

```bash
python agent_examples.py
```

## Integration with Existing Code

The agent is automatically created in `main.py` after training. You can:

1. **Use it directly** in `main.py`:
```python
result = ppd_agent.predict(...)
```

2. **Import it** in other scripts:
```python
from ppd_agent import PPDAgent
agent = PPDAgent.load("ppd_agent.pkl")
```

3. **Start API server**:
```python
from api_server import initialize_agent, app
import uvicorn

initialize_agent(ppd_agent)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Notes

- The agent requires a trained pipeline and training data (for SHAP explainer)
- All categorical features are automatically handled
- SHAP explanations provide interpretability
- The agent can be saved/loaded for deployment

## Support

For issues or questions, refer to the main project documentation or check the example scripts.

