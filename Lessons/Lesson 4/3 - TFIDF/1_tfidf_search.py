import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "docs.json")

# Load docs from JSON file
with open(json_path, "r") as f:
    data = json.load(f)
    docs = data["docs"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

query = "good hummus"

# Transform the query into the same TF-IDF space
query_vec = vectorizer.transform([query])

# Compute cosine similarity between query and each document
similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

# Create a DataFrame with similarity scores
results = pd.DataFrame({"document": docs, "similarity": similarities}).sort_values(
    by="similarity", ascending=False
)

print(results)
