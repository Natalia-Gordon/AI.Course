import numpy as np
from utils.model_cache import get_model

# Load the pre-trained model
model = get_model("glove-wiki-gigaword-100")


# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def print_cosine_diff(word1: str, word2: str):
    vec1, vec2 = model[word1], model[word2]
    similarity = cosine_similarity(vec1, vec2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")


print_cosine_diff("king", "queen")
print_cosine_diff("big", "large")
print_cosine_diff("king", "large")
print_cosine_diff("apple", "orange")
