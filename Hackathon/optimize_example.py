"""
Example script demonstrating how to use RandomizedSearchCV 
for XGBoost hyperparameter optimization.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from MLmodel import optimize_hyperparameters, train_and_evaluate

print("="*60)
print("XGBoost Hyperparameter Optimization Example")
print("="*60)

# 🗂 Load the dataset
df = pd.read_csv("data/postpartum-depression.csv")
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, inplace=True)

# 🧩 Create composite target (same as main.py)
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

threshold = 4
no_threshold = 4
target = "PPD_Composite"
df[target] = ((df['symptom_count'] >= threshold) | 
              (df['no_count'] < no_threshold) | 
              (df['Suicide attempt'] != "No")).astype(int)

# Identify categorical features
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [target, 'symptom_count', 'no_count']]
df.drop(columns=['symptom_count', 'no_count'], axis=1, inplace=True, errors='ignore')
df = df.dropna()

X = df.drop(columns=[target])
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"📊 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 🔍 Optimize hyperparameters
# You can adjust these parameters:
# - n_iter: Number of random combinations to try (default: 50, increase for better results but slower)
# - cv: Cross-validation folds (default: 5)
# - scoring: Metric to optimize (default: 'roc_auc')
best_pipeline, best_params, cv_results = optimize_hyperparameters(
    X_train, y_train, cat_cols,
    n_iter=50,  # Try 50 random combinations (increase to 100+ for better results)
    cv=5,       # 5-fold cross-validation
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1   # Use all available CPU cores
)

# 📊 Evaluate the optimized model on test set
print("\n" + "="*60)
print("📊 Evaluating Optimized Model on Test Set")
print("="*60)
y_proba, y_pred, roc_auc = train_and_evaluate(
    best_pipeline, X_train, y_train, X_test, y_test
)

print(f"\n✅ Final Test ROC AUC: {roc_auc:.4f}")

