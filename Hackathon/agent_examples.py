"""
Example Usage Scripts for PPD Agent Tool

This file demonstrates various ways to use the PPD agent:
1. Standalone Python usage
2. API usage
3. LangChain integration
4. OpenAI Function Calling
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppd_agent import PPDAgent, create_agent_from_training
from langchain_tool import create_langchain_tool
import pandas as pd
from sklearn.model_selection import train_test_split
from MLmodel import create_XGBoost_pipeline, train_and_evaluate


def setup_agent():
    """Setup and return a trained PPD agent."""
    print("Setting up PPD Agent...")
    
    # Load data
    df = pd.read_csv("data/postpartum-depression.csv")
    df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
    df = df.dropna()
    
    # Create target (simplified for example)
    symptom_cols = [
        "Feeling sad or Tearful",
        "Irritable towards baby & partner",
        "Trouble sleeping at night",
        "Problems concentrating or making decision",
        "Overeating or loss of appetite",
        "Feeling anxious",
        "Feeling of guilt",
        "Problems of bonding with baby",
        "Suicide attempt"
    ]
    
    df['symptom_count'] = df[symptom_cols].apply(
        lambda x: (x == "Yes").sum(), axis=1
    )
    df['no_count'] = df[symptom_cols].apply(
        lambda x: (x == "No").sum(), axis=1
    )
    
    target = "PPD_Composite"
    df[target] = ((df['symptom_count'] >= 4) | 
                  (df['no_count'] < 4) | 
                  (df['Suicide attempt'] != "No")).astype(int)
    
    df.drop(columns=['symptom_count', 'no_count'], axis=1, inplace=True, errors='ignore')
    df = df.dropna()
    
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target]
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    pipeline = create_XGBoost_pipeline(cat_cols)
    y_proba, y_pred, roc_auc = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )
    
    # Create agent
    agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
    
    return agent


def example_1_standalone_usage():
    """Example 1: Standalone Python usage."""
    print("\n" + "="*60)
    print("Example 1: Standalone Python Usage")
    print("="*60)
    
    agent = setup_agent()
    
    # Single prediction
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
    
    print("\nPrediction Result:")
    print(f"Risk Score: {result['risk_percentage']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Explanation: {result['explanation']}")
    print(f"\nTop Features:")
    for feature in result['feature_importance'][:3]:
        print(f"  - {feature['feature']}: {feature['impact']} risk")


def example_2_dict_usage():
    """Example 2: Using dictionary input."""
    print("\n" + "="*60)
    print("Example 2: Dictionary Input Usage")
    print("="*60)
    
    agent = setup_agent()
    
    # Predict from dictionary
    patient_data = {
        "Age": "25-30",
        "Feeling sad or Tearful": "No",
        "Irritable towards baby & partner": "No",
        "Trouble sleeping at night": "No",
        "Problems concentrating or making decision": "No",
        "Overeating or loss of appetite": "No",
        "Feeling anxious": "No",
        "Feeling of guilt": "No",
        "Problems of bonding with baby": "No",
        "Suicide attempt": "No"
    }
    
    result = agent.predict_from_dict(patient_data)
    print(f"\nRisk Score: {result['risk_percentage']}%")
    print(f"Risk Level: {result['risk_level']}")


def example_3_batch_prediction():
    """Example 3: Batch predictions."""
    print("\n" + "="*60)
    print("Example 3: Batch Predictions")
    print("="*60)
    
    agent = setup_agent()
    
    # Multiple patients
    patients = [
        {
            "Age": "30-35",
            "Feeling sad or Tearful": "Yes",
            "Irritable towards baby & partner": "Yes",
            "Trouble sleeping at night": "Yes",
            "Problems concentrating or making decision": "Yes",
            "Overeating or loss of appetite": "No",
            "Feeling anxious": "Yes",
            "Feeling of guilt": "Yes",
            "Problems of bonding with baby": "Sometimes",
            "Suicide attempt": "No"
        },
        {
            "Age": "25-30",
            "Feeling sad or Tearful": "No",
            "Irritable towards baby & partner": "No",
            "Trouble sleeping at night": "No",
            "Problems concentrating or making decision": "No",
            "Overeating or loss of appetite": "No",
            "Feeling anxious": "No",
            "Feeling of guilt": "No",
            "Problems of bonding with baby": "No",
            "Suicide attempt": "No"
        }
    ]
    
    results = agent.batch_predict(patients)
    
    print("\nBatch Prediction Results:")
    for i, result in enumerate(results, 1):
        print(f"\nPatient {i}:")
        print(f"  Risk: {result['risk_percentage']}% ({result['risk_level']})")


def example_4_api_usage():
    """Example 4: API usage (requires API server to be running)."""
    print("\n" + "="*60)
    print("Example 4: API Usage")
    print("="*60)
    print("\nTo use the API:")
    print("1. Start the API server: python api_server.py")
    print("2. Make HTTP requests:")
    print("""
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
    """)

# שהוא יכול לענות על תשאולים של מכולה על המידע שנלמד דרך אלגוריתם ML
def example_5_langchain_usage():
    """Example 5: LangChain integration."""
    print("\n" + "="*60)
    print("Example 5: LangChain Integration")
    print("="*60)
    print("\nTo use with LangChain:")
    print("""
    from langchain.agents import initialize_agent, AgentType
    from langchain.llms import OpenAI
    from langchain_tool import create_langchain_tool
    
    # Setup agent
    agent = setup_agent()
    tool = create_langchain_tool(agent)
    
    # Create LangChain agent
    llm = OpenAI(temperature=0)
    langchain_agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Use the agent
    result = langchain_agent.run(
        "What is the PPD risk for a 30-year-old patient "
        "who feels sad, is irritable, and has trouble sleeping?"
    )
    """)


def example_6_openai_function_calling():
    """Example 6: OpenAI Function Calling schema."""
    print("\n" + "="*60)
    print("Example 6: OpenAI Function Calling")
    print("="*60)
    
    agent = setup_agent()
    schema = agent.get_tool_schema()
    
    print("\nTool Schema (for OpenAI Function Calling):")
    import json
    print(json.dumps(schema, indent=2))
    
    print("\n\nTo use with OpenAI:")
    print("""
    import openai
    
    # Get schema
    schema = agent.get_tool_schema()
    
    # Use with OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Assess PPD risk for..."}],
        functions=[schema],
        function_call={"name": "predict_ppd_risk"}
    )
    """)


def example_7_save_load():
    """Example 7: Save and load agent."""
    print("\n" + "="*60)
    print("Example 7: Save and Load Agent")
    print("="*60)
    
    agent = setup_agent()
    
    # Save agent
    import os
    os.makedirs("output/agents", exist_ok=True)
    agent_path = "output/agents/ppd_agent.pkl"
    agent.save(agent_path)
    print(f"✅ Agent saved to {agent_path}")
    
    # Load agent
    loaded_agent = PPDAgent.load(agent_path)
    
    # Use loaded agent
    result = loaded_agent.predict(
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
    
    print(f"\nLoaded agent prediction: {result['risk_percentage']}%")


def display_menu():
    """Display the example selection menu."""
    print("\n" + "="*60)
    print("PPD Agent Tool - Example Usage")
    print("="*60)
    print("\nAvailable Examples:")
    print("  1. Standalone Python Usage")
    print("  2. Dictionary Input Usage")
    print("  3. Batch Predictions")
    print("  4. API Usage (instructions)")
    print("  5. LangChain Integration (instructions)")
    print("  6. OpenAI Function Calling")
    print("  7. Save and Load Agent")
    print("  8. Run All Examples")
    print("  0. Exit")
    print("="*60)


def get_user_choice():
    """Get user's choice from the menu."""
    while True:
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number between 0 and 8.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return 0
        except Exception as e:
            print(f"Error: {e}. Please try again.")


if __name__ == "__main__":
    examples = {
        1: ("Standalone Python Usage", example_1_standalone_usage),
        2: ("Dictionary Input Usage", example_2_dict_usage),
        3: ("Batch Predictions", example_3_batch_prediction),
        4: ("API Usage", example_4_api_usage),
        5: ("LangChain Integration", example_5_langchain_usage),
        6: ("OpenAI Function Calling", example_6_openai_function_calling),
        7: ("Save and Load Agent", example_7_save_load),
    }
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == 0:
            print("\nGoodbye!")
            break
        elif choice == 8:
            print("\n" + "="*60)
            print("Running All Examples")
            print("="*60)
            for num, (name, func) in examples.items():
                print(f"\n>>> Running Example {num}: {name}")
                try:
                    func()
                except Exception as e:
                    print(f"Error running example {num}: {e}")
            print("\n" + "="*60)
            print("All examples completed!")
            print("="*60)
        elif choice in examples:
            name, func = examples[choice]
            print(f"\n>>> Running Example {choice}: {name}")
            try:
                func()
                print("\n" + "="*60)
                print(f"Example {choice} completed!")
                print("="*60)
            except Exception as e:
                print(f"\nError running example: {e}")
        else:
            print("Invalid choice. Please try again.")
        
        # Ask if user wants to continue
        if choice != 0:
            continue_choice = input("\nWould you like to run another example? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("\nGoodbye!")
                break

