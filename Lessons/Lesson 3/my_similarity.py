import gensim.downloader as api

def word_similarity(word1, word2, model):
    """
    Calculates the similarity between two words using the word2vec-google-news-300 model.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.
        model: The pre-loaded word2vec model.

    Returns:
        float: The similarity score between the two words.
    """
    # Get the similarity between the two words
    similarity = model.similarity(word1, word2)

    return similarity

def word_cosine_difference(word1, word2, model):
    """
    Calculates the cosine difference between two words.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.
        model: The pre-loaded word2vec model.

    Returns:
        float: The cosine difference between the two words.
    """
    return 1 - model.similarity(word1, word2)

def find_analogous_word(word1, word2, word3, model):
    """
    Finds the word that completes the analogy "word1 is to word2 as word3 is to ?".

    Args:
        word1 (str): The first word in the analogy base.
        word2 (str): The second word in the analogy base.
        word3 (str): The word to find the analogue for.
        model: The pre-loaded word2vec model.

    Returns:
        list: A list of tuples, where each tuple contains the analogous word and its similarity score.
    """
    return model.most_similar(positive=[word3, word2], negative=[word1])

def print_analogy_matrix(result, word1, word2, word3):
    """
    Prints the full list of analogy results with an explanation.

    Args:
        result (list): A list of (word, score) tuples from find_analogous_word.
        word1 (str): The first word in the analogy base.
        word2 (str): The second word in the analogy base.
        word3 (str): The word to find the analogue for.
    """
    print(f"\nFull analogy results for: '{word1}' is to '{word2}' as '{word3}' is to ?")
    print("Potential answers and their similarity scores:")
    for word, score in result:
        print(f"  {word}: {score:.4f}")

if __name__ == "__main__":
    # Load the pre-trained model (this will download it if not already downloaded)
    #model = api.load("word2vec-google-news-300")
    model = api.load("glove-twitter-25")
    
    # Example usage
    word1 = "king"
    word2 = "queen"
    similarity_score = word_similarity(word1, word2, model)
    print(f"The similarity between '{word1}' and '{word2}' is: {similarity_score}")

    word3 = "man"
    word4 = "woman"
    similarity_score = word_similarity(word3, word4, model)
    print(f"The similarity between '{word3}' and '{word4}' is: {similarity_score}")

    word5 = "cat"
    word6 = "dog"
    similarity_score = word_similarity(word5, word6, model)
    print(f"The similarity between '{word5}' and '{word6}' is: {similarity_score}")

    # Example for cosine difference
    diff_score = word_cosine_difference(word1, word2, model)
    print(f"The cosine difference between '{word1}' and '{word2}' is: {diff_score}")

    # Example for word analogy
    analogy_result = find_analogous_word("man", "woman", "king", model)
    print(f"man is to woman as king is to: {analogy_result[0][0]}")
    print_analogy_matrix(analogy_result, "man", "woman", "king")
