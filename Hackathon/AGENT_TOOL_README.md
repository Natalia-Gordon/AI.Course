# PPD Prediction Agent Tool

This document describes how to use the Postpartum Depression (PPD) Prediction system as an agent tool.

## Overview

The PPD Agent Tool provides multiple interfaces for integrating the PPD prediction model into your applications:

1. **Standalone Python Agent** - Direct Python API
2. **REST API Server** - HTTP API for web services
3. **LangChain Tool** - Integration with LangChain agents
4. **OpenAI Function Calling** - Schema for OpenAI agents
5. **Gradio Interface** - Web UI (existing)

## Available Tools

The PPD Agent provides three main tools:

1. **`predict_ppd_risk`** - Predicts postpartum depression risk based on patient symptoms and demographics
2. **`train_xgboost`** - Trains an XGBoost model for PPD prediction (updates the agent's pipeline)
3. **`train_random_forest`** - Trains a Random Forest model for PPD prediction (updates the agent's pipeline)

## Quick Start

### 1. Standalone Python Usage

```python
from ppd_agent import PPDAgent, create_agent_from_training
from MLmodel import create_XGBoost_pipeline, train_and_evaluate
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

# Get all tool schemas (both prediction and training)
agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
schemas = agent.get_all_tool_schemas()

# Use with OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Assess PPD risk for a patient..."}],
    functions=schemas,
    function_call="auto"  # Let the model choose which tool to use
)
```

### 5. Training Models with Agent Tool

The agent includes tools to train both XGBoost and Random Forest models, which can be useful for:
- Comparing different algorithms (XGBoost vs Random Forest)
- Retraining with different hyperparameters
- Updating the model with new data
- Using hyperparameter optimization

#### Training XGBoost Model

```python
from ppd_agent import PPDAgent
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your agent
agent = PPDAgent.load("ppd_agent.pkl")

# Prepare training data
df = pd.read_csv("data/postpartum-depression.csv")
# ... prepare your data and create X_train, y_train, X_test, y_test ...

# Train XGBoost model with default parameters
result = agent.train_xgboost(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    use_optimization=False,  # Set to True for hyperparameter optimization
    random_state=42
)

print(result["message"])  # "XGBoost model trained successfully! ROC AUC: 0.xxxx"
print(f"ROC AUC Score: {result['roc_auc']:.4f}")

# Now the agent uses the XGBoost model for predictions
prediction = agent.predict(
    age="30-35",
    feeling_sad="Yes",
    # ... other parameters
)
```

**Training XGBoost with Hyperparameter Optimization:**

```python
# Train with RandomizedSearchCV optimization
result = agent.train_xgboost(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    use_optimization=True,   # Enable optimization
    n_iter=30,               # Number of optimization iterations
    cv=3,                    # Cross-validation folds
    scoring='roc_auc',        # Scoring metric
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

print(f"Best Parameters: {result['parameters']}")
print(f"ROC AUC Score: {result['roc_auc']:.4f}")
```

#### Training Random Forest Model

```python
# Train Random Forest model with default parameters
result = agent.train_random_forest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    random_state=42
)

print(result["message"])  # "Random Forest model trained successfully! ROC AUC: 0.xxxx"
print(f"ROC AUC Score: {result['roc_auc']:.4f}")
print(f"Model Parameters: {result['parameters']}")
```

**Training with Custom Hyperparameters:**

```python
# Train with custom Random Forest parameters
result = agent.train_random_forest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_estimators=200,        # Number of trees
    max_depth=15,            # Maximum tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features="sqrt",     # Features to consider: "sqrt", "log2", "auto", or "None"
    test_size=0.2,           # Test set proportion (if not providing X_test/y_test)
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)
```

**Training Response Format:**

Both `train_xgboost()` and `train_random_forest()` return a comprehensive dictionary:

```python
{
    "success": True,
    "roc_auc": 0.8542,
    "message": "XGBoost model trained successfully! ROC AUC: 0.8542",
    "model_type": "XGBoost",  # or "RandomForest"
    "optimization_info": "Hyperparameter optimization applied",  # or "Using default hyperparameters"
    "parameters": {
        # Best hyperparameters (if optimization used) or default parameters
    },
    "pipeline": <sklearn.pipeline.Pipeline>,  # Trained pipeline
    "X_train": <pd.DataFrame>,                # Training features
    "X_test": <pd.DataFrame>,                 # Test features
    "y_train": <pd.Series>,                   # Training labels
    "y_test": <pd.Series>,                    # Test labels
    "y_proba": <np.ndarray>,                  # Prediction probabilities
    "y_pred": <np.ndarray>                    # Binary predictions
}
```

**Note:** After training, the agent's pipeline is automatically updated, and the SHAP explainer is reinitialized. All training results are returned for immediate use in visualizations or further analysis.

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

### Prediction Methods

#### `predict(...)`
Make a single prediction with individual parameters.

#### `predict_from_dict(input_dict)`
Make a prediction from a dictionary of inputs.

#### `batch_predict(input_list)`
Make predictions for multiple patients at once.

### Training Methods

#### `train_xgboost(X_train, y_train, **kwargs)`
Train an XGBoost model and update the agent's pipeline.

**Parameters:**
- `X_train` (pd.DataFrame, required): Training features
- `y_train` (pd.Series, required): Training labels
- `X_test` (pd.DataFrame, optional): Test features (will split if not provided)
- `y_test` (pd.Series, optional): Test labels (will split if not provided)
- `test_size` (float, default=0.2): Test set proportion if splitting
- `random_state` (int, default=42): Random seed for reproducibility
- `use_optimization` (bool, default=False): Whether to use RandomizedSearchCV for hyperparameter optimization
- `n_iter` (int, default=30): Number of optimization iterations (if use_optimization=True)
- `cv` (int, default=3): Number of cross-validation folds (if use_optimization=True)
- `scoring` (str, default='roc_auc'): Scoring metric for optimization
- `n_jobs` (int, default=-1): Number of parallel jobs (-1 uses all cores)

**Returns:**
Comprehensive dictionary with training results including:
- `success`: Whether training was successful
- `roc_auc`: ROC AUC score on test set
- `message`: Status message
- `model_type`: "XGBoost"
- `optimization_info`: Information about optimization
- `parameters`: Best hyperparameters (if optimization used)
- `pipeline`: Trained sklearn pipeline
- `X_train`, `X_test`, `y_train`, `y_test`: Training and test data
- `y_proba`, `y_pred`: Prediction probabilities and binary predictions

**Note:** After training, the agent's pipeline is updated with the XGBoost model, and the SHAP explainer is automatically reinitialized.

#### `train_random_forest(X_train, y_train, **kwargs)`
Train a Random Forest model and update the agent's pipeline.

**Parameters:**
- `X_train` (pd.DataFrame, required): Training features
- `y_train` (pd.Series, required): Training labels
- `X_test` (pd.DataFrame, optional): Test features (will split if not provided)
- `y_test` (pd.Series, optional): Test labels (will split if not provided)
- `test_size` (float, default=0.2): Test set proportion if splitting
- `random_state` (int, default=42): Random seed for reproducibility
- `n_estimators` (int, default=100): Number of trees in the forest
- `max_depth` (int, default=10): Maximum depth of trees
- `min_samples_split` (int, default=2): Minimum samples to split a node
- `min_samples_leaf` (int, default=1): Minimum samples in a leaf node
- `max_features` (str, default="sqrt"): Features to consider: "sqrt", "log2", "auto", or "None"
- `n_jobs` (int, default=-1): Number of parallel jobs (-1 uses all cores)

**Returns:**
Comprehensive dictionary with training results including:
- `success`: Whether training was successful
- `roc_auc`: ROC AUC score on test set
- `message`: Status message
- `model_type`: "RandomForest"
- `parameters`: Model hyperparameters
- `pipeline`: Trained sklearn pipeline
- `X_train`, `X_test`, `y_train`, `y_test`: Training and test data
- `y_proba`, `y_pred`: Prediction probabilities and binary predictions

**Note:** After training, the agent's pipeline is updated with the Random Forest model, and the SHAP explainer is automatically reinitialized.

### Schema Methods

#### `get_tool_schema()`
Get OpenAI Function Calling schema for the prediction tool.

#### `get_training_tool_schema()`
Get OpenAI Function Calling schema for the Random Forest training tool.

**Note:** Currently, only the Random Forest training tool has an OpenAI schema. XGBoost training can be done programmatically using `train_xgboost()`.

#### `get_all_tool_schemas()`
Get all tool schemas (both prediction and training) as a list.

### Utility Methods

#### `save(filepath)`
Save the agent to a file.

#### `load(filepath)`
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

The agent is automatically created in `main.py` after training and is integrated into the Gradio interface. You can:

1. **Use it directly** in `main.py`:
```python
result = ppd_agent.predict(...)
```

2. **Use in Gradio Interface**:
The Gradio interface (`gradio_app.py`) now uses the agent tool for:
- **Predictions**: All predictions go through `agent.predict()` (Standalone Python usage pattern)
- **Training**: Both XGBoost and Random Forest training use `agent.train_xgboost()` and `agent.train_random_forest()` instead of direct algorithm calls
- **SHAP Explanations**: Enhanced SHAP explanations with detailed breakdowns and waterfall visualizations

3. **Import it** in other scripts:
```python
from ppd_agent import PPDAgent
agent = PPDAgent.load("ppd_agent.pkl")
```

4. **Start API server**:
```python
from api_server import initialize_agent, app
import uvicorn

initialize_agent(ppd_agent)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Notes

- The agent requires a trained pipeline and training data (for SHAP explainer)
- All categorical features are automatically handled
- SHAP explanations provide interpretability with detailed breakdowns and visualizations
- The agent can be saved/loaded for deployment
- The agent supports both XGBoost (default) and Random Forest models
- You can switch between models by calling `train_xgboost()` or `train_random_forest()` to update the pipeline
- After training, the agent automatically updates its pipeline and SHAP explainer
- Both training methods return comprehensive results including predictions for immediate use in visualizations
- The Gradio interface uses the agent tool for all operations (predictions and training), providing a consistent interface
- Enhanced SHAP explanations in Gradio include:
  - Detailed feature contribution breakdowns
  - Impact levels (high/moderate/low)
  - Waterfall visualization showing cumulative effects
  - Bar plots showing individual feature contributions

## Support

For issues or questions, refer to the main project documentation or check the example scripts.

