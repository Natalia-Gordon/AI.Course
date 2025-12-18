# 📌 Standard libraries
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from MLmodel import create_pipeline, train_and_evaluate
from visualization import create_all_visualizations
from gradio_app import create_gradio_interface

print("Welcome to the Postpartum Depression Prediction System")

# 🗂 Load the PostPartum Depression dataset CSV (download from Kaggle)
df = pd.read_csv("data/postpartum-depression.csv")

# 📊 Show basic info
print(df.shape)
print(df.head())
print(df.info())

# Drop the Timestamp column
df.drop(columns=['Timestamp'], axis=1, inplace=True)

# Check the unique values in each column of data
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

# Drop the rows with missing values
df.dropna(axis=0, inplace=True)

# 🧩 Create composite target based on multiple symptoms (better for PPD diagnosis)
# Define all symptom columns
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

# Calculate symptom count (how many symptoms answered "Yes")
df['symptom_count'] = df[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)

# Calculate "No" answer count (how many symptoms answered "No")
df['no_count'] = df[symptom_cols].apply(
    lambda x: (x == "No").sum(), axis=1
)

# Create composite target: PPD = 1 if:
# - 4+ symptoms present, OR
# - less than 4 "No" answers, OR
# - "Suicide attempt" is not "No" (very high risk indicator)
# If someone has very few "No" answers, they're experiencing many symptoms, indicating higher risk
threshold = 4
no_threshold = 4
target = "PPD_Composite"
df[target] = ((df['symptom_count'] >= threshold) | 
              (df['no_count'] < no_threshold) | 
              (df['Suicide attempt'] != "No")).astype(int)

print(f"\n📊 Composite Target Distribution (PPD = 1 if {threshold}+ 'Yes' symptoms OR <{no_threshold} 'No' answers OR Suicide attempt != 'No'):")
print(df[target].value_counts())
print(f"Proportions: {df[target].value_counts(normalize=True).to_dict()}")

# 🧩 Identify categorical features (all symptom columns + Age, excluding target columns)
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [target, 'symptom_count', 'no_count']]

# 🧪 Handle missing values (drop symptom_count and no_count columns as they're just for target creation)
df.drop(columns=['symptom_count', 'no_count'], axis=1, inplace=True, errors='ignore')
df = df.dropna()

X = df.drop(columns=[target])
y = df[target]

print("Final feature shape:", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 🧠 Create and train the model using MLmodel.py
print("\n" + "="*50)
print("Training XGBoost Model...")
print("="*50)

pipeline = create_pipeline(cat_cols)
y_proba, y_pred, roc_auc = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

# 📊 Create visualizations
create_all_visualizations(df, X_test, y_test, y_pred, y_proba, roc_auc, 
                         pipeline, cat_cols, target)

# 🚀 Launch Gradio Interface
print("\n" + "="*50)
print("Launching Gradio Web Interface...")
print("="*50)

# Create Gradio interface (examples are already included in the interface)
interface = create_gradio_interface(pipeline, X_train, cat_cols)

print("\n✅ Gradio interface is ready!")
print("📱 The web interface will open in your browser.")
print("💡 You can use the example cases below the form for quick testing.")
print("📊 Example cases include:")
print("   - High risk case (multiple symptoms)")
print("   - Low risk case (no symptoms)")
print("   - Moderate risk case (some symptoms)")
print("   - Very high risk case (all symptoms)")
print("   - Low-moderate risk (sleep issues only)")
print("\n" + "="*50)

# Launch the interface
interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
