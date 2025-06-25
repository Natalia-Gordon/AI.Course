import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "chatGpT is helpfull assistant",
    "chatGPT is a large language model.",
    "chatGPT helps with natural language processing.",
    "Artifitial intelligence is powerfull."
]

vectorizer = TfidfVectorizer()

# Generate the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(docs)

# Create a pandas DataFrame to display the TF-IDF matrix
# The rows are the documents, and the columns are the words from the vocabulary.
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=docs, columns=vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_df)

# Print the feature names (vocabulary)
print("\nFeature names (Vocabulary):")
print(vectorizer.get_feature_names_out())

# Print the TF-IDF matrix
print("\nTF-IDF Vector Matrix (sparse):")
print(tfidf_matrix) 

# Print the TF-IDF matrix as a dense array for better readability
print("\nTF-IDF Vector Matrix (dense):")
print(tfidf_matrix.toarray()) 

