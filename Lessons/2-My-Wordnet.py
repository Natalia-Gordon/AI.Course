import nltk
from nltk.corpus import wordnet

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

def print_synonyms(word):
    """
    Prints synonyms of a word using WordNet.
    """

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if synonyms:
        print(f"Synonyms for '{word}': {', '.join(synonyms)}")
        return True
    else:
        print(f"No synonyms found for '{word}'.")
        return False


def print_hypernyms(word):
    """
    Prints hypernyms of a word using WordNet.
    """
    hypernyms = set()
    for syn in wordnet.synsets(word):
        for hyper in syn.hypernyms():
            for l in hyper.lemmas():
                hypernyms.add(l.name())
    if hypernyms:
        print(f"Hypernyms for '{word}': {', '.join(hypernyms)}")
        return True
    else:
        print(f"No hypernyms found for '{word}'.")
        return False

# main
print_synonyms("good")
print_hypernyms("chair")