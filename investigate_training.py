"""
Investigate the training process to understand model behavior.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Try to import model, but continue without it if not available
try:
    from MLmodel import create_pipeline, train_and_evaluate
    MODEL_AVAILABLE = True
except ImportError:
    print("Warning: Cannot import MLmodel (xgboost not installed). Will analyze data only.")
    MODEL_AVAILABLE = False

print("="*60)
print("INVESTIGATING TRAINING PROCESS")
print("="*60)

# Load data
print("\n1. Loading and examining data...")
df = pd.read_csv("data/postpartum-depression.csv")
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, inplace=True)

# Check target variable
target = "Feeling anxious"
print(f"\n2. Target variable '{target}' distribution:")
print(df[target].value_counts())
print(f"\nTarget variable proportions:")
print(df[target].value_counts(normalize=True))

# Encode target
df[target] = df[target].map({"Yes": 1, "No": 0})
print(f"\n3. After encoding (Yes=1, No=0):")
print(df[target].value_counts())
print(f"Proportions: {df[target].value_counts(normalize=True).to_dict()}")

# Prepare features
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target]
X = df.drop(columns=[target])
y = df[target]

print(f"\n4. Feature columns: {list(X.columns)}")
print(f"   Categorical columns: {cat_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n5. Training set target distribution:")
print(f"   Class 0 (No): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"   Class 1 (Yes): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"\n6. Test set target distribution:")
print(f"   Class 0 (No): {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"   Class 1 (Yes): {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

# Train model (if available)
if MODEL_AVAILABLE:
    print("\n7. Training model...")
    pipeline = create_pipeline(cat_cols)
    y_proba, y_pred, roc_auc = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    
    # Check model classes
    model = pipeline.named_steps['model']
    print(f"\n8. Model classes: {model.classes_}")
    print(f"   Class 0 = {model.classes_[0]} (No depression)")
    print(f"   Class 1 = {model.classes_[1]} (Yes depression)")
    
    # Test with example cases
    print("\n9. Testing with example cases:")
    print("-" * 60)
    
    examples = [
        {
            "name": "Example 1 (High risk - 7 Yes)",
            "data": {
                "Age": 28,
                "Feeling sad or Tearful": "Yes",
                "Irritable towards baby & partner": "Yes",
                "Trouble sleeping at night": "Yes",
                "Problems concentrating or making decision": "Yes",
                "Overeating or loss of appetite": "Yes",
                "Feeling of guilt": "Yes",
                "Problems of bonding with baby": "Yes",
                "Suicide attempt": "No"
            }
        },
        {
            "name": "Example 2 (Low risk - all No)",
            "data": {
                "Age": 32,
                "Feeling sad or Tearful": "No",
                "Irritable towards baby & partner": "No",
                "Trouble sleeping at night": "No",
                "Problems concentrating or making decision": "No",
                "Overeating or loss of appetite": "No",
                "Feeling of guilt": "No",
                "Problems of bonding with baby": "No",
                "Suicide attempt": "No"
            }
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        row = pd.DataFrame([example['data']], columns=X_train.columns)
        
        # Ensure dtypes match
        for col in row.columns:
            if col in X_train.dtypes:
                target_dtype = X_train.dtypes[col]
                if target_dtype == 'object':
                    row[col] = row[col].astype(str)
                elif col == "Age":
                    row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0.0)
                    if 'int' in str(target_dtype):
                        row[col] = row[col].astype(int)
                    else:
                        row[col] = row[col].astype(float)
        
        proba_result = pipeline.predict_proba(row)
        pred_class = pipeline.predict(row)[0]
        
        prob_class_0 = proba_result[0, 0]
        prob_class_1 = proba_result[0, 1]
        
        print(f"  Input: {example['data']}")
        print(f"  Prob class 0 (No depression): {prob_class_0:.4f} ({prob_class_0*100:.2f}%)")
        print(f"  Prob class 1 (Yes depression): {prob_class_1:.4f} ({prob_class_1*100:.2f}%)")
        print(f"  Predicted class: {pred_class} ({'No depression' if pred_class == 0 else 'Yes depression'})")
        print(f"  Using prob_class_1: {prob_class_1*100:.2f}%")
else:
    print("\n7. Skipping model training (xgboost not available)")

# Check training data patterns
print("\n10. Analyzing training data patterns:")
print("-" * 60)

# Count symptoms for each case
symptom_cols = [c for c in X_train.columns if c != "Age"]
X_train_with_counts = X_train.copy()
X_train_with_counts['symptom_count'] = X_train[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)
X_train_with_counts['target'] = y_train

print("\nSymptom count distribution by target:")
symptom_patterns = {}
for symptom_count in sorted(X_train_with_counts['symptom_count'].unique()):
    subset = X_train_with_counts[X_train_with_counts['symptom_count'] == symptom_count]
    if len(subset) > 0:
        yes_count = sum(subset['target'] == 1)
        no_count = sum(subset['target'] == 0)
        yes_pct = yes_count / len(subset) * 100 if len(subset) > 0 else 0
        symptom_patterns[symptom_count] = yes_pct / 100.0  # Store as probability
        print(f"  {symptom_count} symptoms: {len(subset)} cases, "
              f"Depression: {yes_count} ({yes_pct:.1f}%), No depression: {no_count} ({100-yes_pct:.1f}%)")

print("\n⚠️  KEY FINDING: The data shows unusual patterns!")
print("   - 1-2 symptoms: Very high depression rate (77-81%)")
print("   - 5 symptoms: Very low depression rate (10%)")
print("   - This suggests data quality issues or collection problems")
print("   - The model is learning these patterns, causing inconsistent predictions")

print("\n" + "="*60)
print("INVESTIGATION COMPLETE")
print("="*60)

