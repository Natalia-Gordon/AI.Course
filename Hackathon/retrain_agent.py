"""
Script to retrain and save the PPD agent with the current sklearn version.
This fixes version incompatibility issues when loading the agent.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from MLmodel import create_XGBoost_pipeline, train_and_evaluate
from ppd_agent import create_agent_from_training

print("=" * 60)
print("Retraining PPD Agent (fixing sklearn version compatibility)")
print("=" * 60)

# Load the data
print("\nLoading data...")
df = pd.read_csv("data/postpartum-depression.csv")

# Drop the Timestamp column
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Create composite target
print("Creating composite target...")
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

# Calculate symptom count
df['symptom_count'] = df[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)

# Calculate "No" answer count
df['no_count'] = df[symptom_cols].apply(
    lambda x: (x == "No").sum(), axis=1
)

# Create composite target
threshold = 4
no_threshold = 4
target = "PPD_Composite"
df[target] = ((df['symptom_count'] >= threshold) | 
              (df['no_count'] < no_threshold) | 
              (df['Suicide attempt'] != "No")).astype(int)

print(f"Target distribution: {df[target].value_counts().to_dict()}")

# Identify categorical features
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [target, 'symptom_count', 'no_count']]

# Drop helper columns
df.drop(columns=['symptom_count', 'no_count'], axis=1, inplace=True, errors='ignore')
df = df.dropna()

# Prepare features and target
X = df.drop(columns=[target])
y = df[target]

print(f"Features shape: {X.shape}")

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Create and train the model
print("\nTraining XGBoost model...")
pipeline = create_XGBoost_pipeline(cat_cols)
y_proba, y_pred, roc_auc = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
print(f"Model trained! ROC-AUC: {roc_auc:.4f}")

# Create and save the agent
print("\nCreating PPD Agent...")
ppd_agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
print("PPD Agent created!")

# Save agent
print("\nSaving agent to ppd_agent.pkl...")
ppd_agent.save("ppd_agent.pkl")
print("Agent saved successfully!")

print("\n" + "=" * 60)
print("Done! The agent is now compatible with your current sklearn version.")
print("You can now restart your API server and the agent should load correctly.")
print("=" * 60)

