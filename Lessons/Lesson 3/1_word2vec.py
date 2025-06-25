import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_cache import get_model

# Load pre-trained word2vec model
model = get_model("word2vec-google-news-300")  # Will use cache if available


def print_embedding(word: str):
    embedding = model[word]
    print(f"{word}: First 10 values of embedding:")
    print(embedding[:10])

print_embedding("big")
print_embedding("large")

print_embedding("king")
print_embedding("queen")

print_embedding("computer")
print_embedding("laptop")
