"""
Explore different target variable options for postpartum depression prediction.
Shows how to create composite targets or use multi-output prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("="*70)
print("EXPLORING ALTERNATIVE TARGET VARIABLE OPTIONS")
print("="*70)

# Load data
df = pd.read_csv("data/postpartum-depression.csv")
df.drop(columns=['Timestamp'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, inplace=True)

# Define symptom columns
symptom_cols = [
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt"
]

print("\n" + "="*70)
print("OPTION 1: COMPOSITE TARGET (Recommended for PPD)")
print("="*70)
print("Create a single target variable based on multiple symptoms")
print("This is clinically more appropriate for PPD diagnosis.\n")

# Option 1a: Symptom count threshold
df['symptom_count'] = df[symptom_cols].apply(
    lambda x: (x == "Yes").sum(), axis=1
)

# Threshold-based composite target (if 3+ symptoms = PPD)
threshold = 3
df['ppd_composite_threshold'] = (df['symptom_count'] >= threshold).astype(int)

print(f"1a. Threshold-based: PPD = 1 if symptom_count >= {threshold}")
print(f"   Distribution:")
print(df['ppd_composite_threshold'].value_counts())
print(f"   Proportions: {df['ppd_composite_threshold'].value_counts(normalize=True).to_dict()}\n")

# Option 1b: Weighted composite (suicide attempt = critical)
df['ppd_composite_weighted'] = 0
df.loc[df['Suicide attempt'] == 'Yes', 'ppd_composite_weighted'] = 1
df.loc[(df['symptom_count'] >= 4) & (df['Suicide attempt'] != 'Yes'), 'ppd_composite_weighted'] = 1

print("1b. Weighted composite: PPD = 1 if (Suicide attempt = Yes) OR (4+ other symptoms)")
print(f"   Distribution:")
print(df['ppd_composite_weighted'].value_counts())
print(f"   Proportions: {df['ppd_composite_weighted'].value_counts(normalize=True).to_dict()}\n")

# Option 1c: Key symptoms only (most clinically relevant)
key_symptoms = [
    "Feeling sad or Tearful",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt"
]
df['key_symptom_count'] = df[key_symptoms].apply(
    lambda x: (x == "Yes").sum(), axis=1
)
df['ppd_key_symptoms'] = (df['key_symptom_count'] >= 2).astype(int)

print("1c. Key symptoms only: PPD = 1 if 2+ of (Sad, Guilt, Bonding issues, Suicide)")
print(f"   Distribution:")
print(df['ppd_key_symptoms'].value_counts())
print(f"   Proportions: {df['ppd_key_symptoms'].value_counts(normalize=True).to_dict()}\n")

print("\n" + "="*70)
print("OPTION 2: MULTI-OUTPUT PREDICTION")
print("="*70)
print("Predict multiple targets simultaneously (requires MultiOutputClassifier)")
print("Could predict: ['Feeling anxious', 'Feeling sad or Tearful', 'Suicide attempt']\n")

# Show what multi-output would look like
multi_targets = ["Feeling anxious", "Feeling sad or Tearful", "Suicide attempt"]
for target in multi_targets:
    print(f"  {target}:")
    print(f"    {df[target].value_counts().to_dict()}")

print("\nNote: This would require using sklearn.multioutput.MultiOutputClassifier")
print("and would give separate predictions for each target.\n")

print("="*70)
print("OPTION 3: USE A DIFFERENT SINGLE TARGET")
print("="*70)
print("Use a different symptom as the target (instead of 'Feeling anxious')\n")

alternative_targets = ["Feeling sad or Tearful", "Suicide attempt", "Problems of bonding with baby"]
for alt_target in alternative_targets:
    print(f"  {alt_target}:")
    counts = df[alt_target].value_counts()
    print(f"    Distribution: {counts.to_dict()}")
    print(f"    Proportions: {counts.value_counts(normalize=True).to_dict()}\n")

print("="*70)
print("RECOMMENDATION")
print("="*70)
print("For PPD prediction, Option 1 (Composite Target) is recommended because:")
print("  1. PPD is a complex condition diagnosed by multiple symptoms")
print("  2. A single symptom (like 'Feeling anxious') may not be sufficient")
print("  3. Composite targets better reflect clinical diagnosis criteria")
print("  4. More robust and medically appropriate")
print("\nSuggested approach:")
print("  - Use symptom_count >= 3 OR key_symptoms >= 2")
print("  - This creates a binary target: PPD (1) vs No PPD (0)")
print("  - Train model to predict this composite target")
print("  - Features would be: Age + all symptom questions")

print("\n" + "="*70)
print("COMPARISON: Current vs Composite Target")
print("="*70)

current_target = "Feeling anxious"
df[current_target + "_encoded"] = df[current_target].map({"Yes": 1, "No": 0})

print(f"\nCurrent target ('{current_target}'):")
print(f"  Class 0: {sum(df[current_target + '_encoded'] == 0)} ({sum(df[current_target + '_encoded'] == 0)/len(df)*100:.1f}%)")
print(f"  Class 1: {sum(df[current_target + '_encoded'] == 1)} ({sum(df[current_target + '_encoded'] == 1)/len(df)*100:.1f}%)")

print(f"\nComposite target (symptom_count >= {threshold}):")
print(f"  Class 0: {sum(df['ppd_composite_threshold'] == 0)} ({sum(df['ppd_composite_threshold'] == 0)/len(df)*100:.1f}%)")
print(f"  Class 1: {sum(df['ppd_composite_threshold'] == 1)} ({sum(df['ppd_composite_threshold'] == 1)/len(df)*100:.1f}%)")

print("\n" + "="*70)

