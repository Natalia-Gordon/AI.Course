from utils.model_cache import get_model

# Load pre-trained GloVe model
model = get_model("glove-wiki-gigaword-100")


def get_nearest_word(source_word: str, target_word: str, query_word: str):
    """
    Find words that complete the analogy: source_word is to target_word as query_word is to X.

    Args:
        source_word: The first word in the source pair (e.g., 'king' in 'king is to queen')
        target_word: The second word in the source pair (e.g., 'queen' in 'king is to queen')
        query_word: The word to find the analogy for (e.g., 'man' in 'as man is to X')
    """
    return model.most_similar(
        positive=[query_word, target_word], negative=[source_word]
    )


def print_nearest_word(source_word: str, target_word: str, query_word: str):
    result = get_nearest_word(source_word, target_word, query_word)
    print(f"Analogy: '{source_word}' is to '{target_word}' as '{query_word}' is to:")
    for word, score in result:
        print(f"{word:<10} similarity: {score:.4f}")


# Examples
print_nearest_word("dog", "puppy", "cat")  # dog:puppy :: cat:?
print_nearest_word("king", "queen", "man")  # king:queen :: man:?
print_nearest_word("france", "paris", "germany")  # france:paris :: germany:?
