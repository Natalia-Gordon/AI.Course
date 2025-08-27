from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfIndexer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None
        self.docs: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]):
        self.docs = chunks
        texts = [c.get('text','') for c in chunks]
        if texts:
            self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.matrix is None:
            return []
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        idx = scores.argsort()[::-1][:k]
        return [{**self.docs[i], 'score': float(scores[i])} for i in idx]