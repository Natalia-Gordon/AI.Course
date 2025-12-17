"""
Analyze how the target variable influences SHAP values and PPD Risk Score.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("="*70)
print("ANALYZING TARGET VARIABLE INFLUENCE ON SHAP AND RISK SCORE")
print("="*70)

# Load and prepare data
df = pd.read_csv("data/postpartum-depression.csv")
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, inplace=True)

target = "Feeling anxious"
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target]

# Check original target values
print("\n1. TARGET VARIABLE ANALYSIS")
print("-" * 70)
print(f"Target variable: '{target}'")
print(f"\nOriginal distribution:")
print(df[target].value_counts())
print(f"\nProportions:")
print(df[target].value_counts(normalize=True))

# Encode target
df[target] = df[target].map({"Yes": 1, "No": 0})
X = df.drop(columns=[target])
y = df[target]

print(f"\n2. TARGET ENCODING")
print("-" * 70)
print("Encoding: 'Yes' = 1 (depression/anxiety), 'No' = 0 (no depression/anxiety)")
print(f"After encoding:")
print(f"  Class 0 (No): {sum(y == 0)} cases ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"  Class 1 (Yes): {sum(y == 1)} cases ({sum(y == 1)/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n3. HOW TARGET AFFECTS MODEL PREDICTIONS")
print("-" * 70)
print("The model learns to predict P(class 1) = P('Feeling anxious' = 'Yes')")
print("This means:")
print("  - prob_class_0 = P(No anxiety/depression)")
print("  - prob_class_1 = P(Yes anxiety/depression) = PPD Risk Score")
print("\n⚠️  IMPORTANT: The target is 'Feeling anxious', not 'Postpartum Depression'")
print("   The model predicts anxiety, which is used as a proxy for PPD risk.")

print(f"\n4. SHAP VALUES EXPLANATION")
print("-" * 70)
print("SHAP values explain how each feature contributes to the prediction.")
print("In the current implementation:")
print("  - SHAP values are calculated for class 1 (Yes anxiety/depression)")
print("  - Positive SHAP value → increases probability of class 1")
print("  - Negative SHAP value → decreases probability of class 1")
print("\nCurrent code uses:")
print("  shap_values = explainer.shap_values(row_processed)")
print("  shap_values = shap_values[1]  # Uses class 1 (Yes depression)")

print(f"\n5. TARGET VARIABLE IMPACT ON PREDICTIONS")
print("-" * 70)

# Analyze feature correlations with target
print("\nFeature correlation with target (Feeling anxious):")
symptom_cols = [c for c in X_train.columns if c != "Age"]

for col in symptom_cols:
    if col in X_train.columns:
        # Create binary encoding for correlation
        X_binary = X_train.copy()
        X_binary[col] = (X_binary[col] == "Yes").astype(int)
        
        # Calculate correlation
        correlation = X_binary[col].corr(y_train)
        print(f"  {col}: {correlation:.3f}")

print(f"\n6. EXAMPLE: How target distribution affects predictions")
print("-" * 70)

# Count symptoms and their relationship to target
symptom_cols = [c for c in X_train.columns if c != "Age"]
X_train_with_counts = X_train.copy()
X_train_with_counts['symptom_count'] = X_train[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)
X_train_with_counts['target'] = y_train

print("\nSymptom count vs Target (Feeling anxious):")
for symptom_count in sorted(X_train_with_counts['symptom_count'].unique()):
    subset = X_train_with_counts[X_train_with_counts['symptom_count'] == symptom_count]
    if len(subset) > 0:
        yes_count = sum(subset['target'] == 1)
        total = len(subset)
        yes_pct = yes_count / total * 100
        print(f"  {symptom_count} symptoms: {yes_pct:.1f}% have 'Feeling anxious' = Yes")
        print(f"    → Model learns: {symptom_count} symptoms → {yes_pct:.1f}% PPD risk")

print(f"\n7. IMPACT ON SHAP VALUES")
print("-" * 70)
print("SHAP values are calculated based on:")
print("  1. The model's learned patterns from training data")
print("  2. The specific input features for a prediction")
print("  3. The target variable distribution (class imbalance)")
print("\nSince the target has:")
print(f"  - Class 0 (No): {sum(y_train == 0)/len(y_train)*100:.1f}%")
print(f"  - Class 1 (Yes): {sum(y_train == 1)/len(y_train)*100:.1f}%")
print("\nThe model is biased toward predicting class 1 (Yes) due to class imbalance.")
print("This affects:")
print("  - Base SHAP values (starting point)")
print("  - Feature importance rankings")
print("  - Individual feature contributions")

print(f"\n8. IMPACT ON PPD RISK SCORE")
print("-" * 70)
print("The PPD Risk Score = prob_class_1 = P('Feeling anxious' = 'Yes')")
print("\nThis is affected by:")
print("  1. Target variable distribution (64.9% Yes, 35.1% No)")
print("     → Model has baseline ~65% probability for class 1")
print("  2. Unusual data patterns (1-2 symptoms = high risk, 5 symptoms = low risk)")
print("     → Model learns counterintuitive relationships")
print("  3. Class imbalance")
print("     → Model tends to predict class 1 more often")

print(f"\n9. RECOMMENDATIONS")
print("-" * 70)
print("To improve SHAP values and risk scores:")
print("  1. Address class imbalance (use class_weight='balanced' or SMOTE)")
print("  2. Fix data quality issues (investigate unusual symptom patterns)")
print("  3. Consider if 'Feeling anxious' is the right target for PPD prediction")
print("  4. Use calibrated probabilities for more accurate risk scores")
print("  5. Calculate SHAP for both classes to see full picture")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

