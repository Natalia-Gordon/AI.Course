import os

import dspy
from dotenv import load_dotenv

from src.vector_store import VectorStore

# --- 1. Setup DSPy --- 

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Configure the language model (DSPy v2+ unified LM interface)
# Note: Use a widely supported Gemini model name. You can change to a newer model if available.
llm = dspy.LM(
    "gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    max_output_tokens=4096, 
)

# Apply configuration
dspy.configure(lm=llm)

INDEX_SAVE_DIR = os.path.join('data', 'moby_dick_index')

# --- 2. Define the Retriever --- 

class FaissRetriever(dspy.Module):
    """A custom retriever that uses our FAISS vector store."""
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        print("Loading FAISS index...")
        self.vector_store = VectorStore()
        self.vector_store.load_index(INDEX_SAVE_DIR)
        print("FAISS index loaded successfully.")

    def forward(self, query_or_queries: str, k: int | None = None) -> dspy.Prediction:
        """
        Search for top k passages for a given query.
        """
        k = k if k is not None else self.k
        # Our vector store search returns a list of strings
        passages = self.vector_store.search(query_or_queries, k=k)
        # DSPy's Retrieve expects a list of dspy.Example objects or dicts
        return dspy.Prediction(passages=passages)

# --- 3. Define the RAG Signature and Module --- 

class GenerateAnswer(dspy.Signature):
    """Answer questions based on the provided context from the book Moby Dick."""
    context = dspy.InputField(desc="Relevant passages from Moby Dick.")
    question = dspy.InputField(desc="The user's question.")
    answer = dspy.OutputField(desc="A comprehensive answer to the question based on the context.")

class RAG(dspy.Module):
    """The main RAG pipeline module."""
    def __init__(self, k: int = 3):
        super().__init__()
        self.retrieve = FaissRetriever(k=k)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        """
        The main forward pass for the RAG pipeline.
        """
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# --- Example Usage --- 
if __name__ == '__main__':
    print("Initializing RAG pipeline...")
    rag_pipeline = RAG(k=3)

    # Ask a question
    question = "What is the Pequod?"
    print(f"\n--- Question ---\n{question}")

    # Get the prediction
    prediction = rag_pipeline(question)

    print(f"\n--- Answer ---\n{prediction.answer}")

    # Inspect the history
    # llm.inspect_history(n=1)
