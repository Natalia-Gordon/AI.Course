import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore.

        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[str]):
        """
        Builds the FAISS index from a list of text chunks.

        Args:
            chunks: A list of text documents or paragraphs.
        """
        print("Encoding text chunks into vectors...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunks = chunks
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Successfully built FAISS index with {self.index.ntotal} vectors.")

    def save_index(self, dir_path: str):
        """
        Saves the FAISS index and the text chunks to a directory.

        Args:
            dir_path: The directory to save the files in.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        faiss.write_index(self.index, os.path.join(dir_path, 'index.faiss'))
        with open(os.path.join(dir_path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Successfully saved index and chunks to {dir_path}")

    def load_index(self, dir_path: str):
        """
        Loads the FAISS index and text chunks from a directory.

        Args:
            dir_path: The directory where the files are saved.
        """
        self.index = faiss.read_index(os.path.join(dir_path, 'index.faiss'))
        with open(os.path.join(dir_path, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Successfully loaded index and chunks from {dir_path}")

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        Searches the index for the top k most similar chunks to the query.

        Args:
            query: The query string.
            k: The number of results to return.

        Returns:
            A list of the top k most similar text chunks.
        """
        if self.index is None:
            raise RuntimeError("Index is not built or loaded. Please build or load an index first.")
        
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, k)
        
        return [self.chunks[i] for i in indices[0]]
