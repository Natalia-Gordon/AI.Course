# 📌 Standard libraries
import pandas as pd
import numpy as np
import shap

print("Welcome to the Postpartum Depression Prediction System")

# 🗂 Load the PostPartum Depression dataset CSV (download from Kaggle)
df = pd.read_csv("data/postpartum-depression.csv")

# 📊 Show basic info
print(df.shape)
print(df.head())
print(df.info())