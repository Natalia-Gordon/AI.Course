import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "ChatGPT is helpful assistant.",
    "ChatGPT is a large language model.",
    "ChatGPT helps with natural language processing.",
    "Artificial intelligence is powerful.",
]

vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(docs)

# Convert to pandas DataFrame for readability
df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()).T
df.columns = [f"doc_{i + 1}" for i in range(len(docs))]

print(df)
