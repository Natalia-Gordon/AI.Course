"""
Test script to verify the prediction output for the first example case.
"""
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MLmodel import create_pipeline, train_and_evaluate
from sklearn.model_selection import train_test_split

# Load data
print("Loading data...")
df = pd.read_csv("data/postpartum-depression.csv")
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, inplace=True)

# Identify categorical features
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "Feeling anxious"]
target = "Feeling anxious"

# Encode target
df[target] = df[target].map({"Yes": 1, "No": 0})
df = df.dropna()

X = df.drop(columns=[target])
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
print("Training model...")
pipeline = create_pipeline(cat_cols)
y_proba, y_pred, roc_auc = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

# Test first example case
print("\n" + "="*50)
print("Testing First Example Case:")
print("="*50)
print("Input values:")
print("  Age: 28")
print("  Feeling sad or Tearful: Yes")
print("  Irritable towards baby & partner: Yes")
print("  Trouble sleeping at night: Yes")
print("  Problems concentrating or making decision: Yes")
print("  Overeating or loss of appetite: Yes")
print("  Feeling of guilt: Yes")
print("  Problems of bonding with baby: Yes")
print("  Suicide attempt: No")
print()

# Create input DataFrame matching the exact structure
feature_columns = list(X_train.columns)
feature_dtypes = X_train.dtypes.to_dict()

row_dict = {}
for col in feature_columns:
    if col == "Age":
        row_dict[col] = 28
    elif col == "Feeling sad or Tearful":
        row_dict[col] = "Yes"
    elif col == "Irritable towards baby & partner":
        row_dict[col] = "Yes"
    elif col == "Trouble sleeping at night":
        row_dict[col] = "Yes"
    elif col == "Problems concentrating or making decision":
        row_dict[col] = "Yes"
    elif col == "Overeating or loss of appetite":
        row_dict[col] = "Yes"
    elif col == "Feeling of guilt":
        row_dict[col] = "Yes"
    elif col == "Problems of bonding with baby":
        row_dict[col] = "Yes"
    elif col == "Suicide attempt":
        row_dict[col] = "No"
    else:
        row_dict[col] = "No"  # Default

row = pd.DataFrame([row_dict], columns=feature_columns)

# Convert dtypes to match training data
for col in row.columns:
    if col in feature_dtypes:
        target_dtype = feature_dtypes[col]
        if target_dtype == 'object':
            row[col] = row[col].astype(str)
        elif col == "Age":
            row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0.0)
            if 'int' in str(target_dtype):
                row[col] = row[col].astype(int)
            else:
                row[col] = row[col].astype(float)

# Get prediction
proba = pipeline.predict_proba(row)[0, 1]
risk_score = f"PPD Risk Score: {proba:.2%}"

print("Prediction Result:")
print(f"  {risk_score}")
print(f"  Probability (raw): {proba:.6f}")
print(f"  Predicted class: {pipeline.predict(row)[0]}")

# Compare with similar cases in training data
print("\n" + "="*50)
print("Comparison with similar cases in training data:")
print("="*50)

# Find cases with at least 6 "Yes" symptoms
symptom_cols = [c for c in X_train.columns if c != "Age"]
train_with_counts = X_train.copy()
train_with_counts['yes_count'] = train_with_counts[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)
train_with_counts['target'] = y_train

# Cases with 6+ symptoms
high_symptom_cases = train_with_counts[train_with_counts['yes_count'] >= 6]
if len(high_symptom_cases) > 0:
    print(f"\nFound {len(high_symptom_cases)} cases with 6+ symptoms")
    print(f"Average probability in similar cases: {high_symptom_cases['target'].mean():.2%}")
    
    # Test a few similar cases
    print("\nTesting a few similar cases from training data:")
    for idx in high_symptom_cases.head(5).index:
        test_row = X_train.loc[[idx]]
        test_proba = pipeline.predict_proba(test_row)[0, 1]
        actual_target = y_train.loc[idx]
        print(f"  Case {idx}: {test_proba:.2%} (actual: {'Yes' if actual_target == 1 else 'No'})")

print("\n✅ Test completed!")

