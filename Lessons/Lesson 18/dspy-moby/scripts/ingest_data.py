import os
import sys

# Add the src directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_chunk_text
from src.vector_store import VectorStore

# Define file paths
DATA_FILE_PATH = os.path.join('data', 'moby_dick', 'moby_dick.txt')
INDEX_SAVE_DIR = os.path.join('data', 'moby_dick_index')

def main():
    """
    Main function to run the data ingestion process.
    """
    print(f"Loading data from {DATA_FILE_PATH}...")
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please ensure the 'data/moby_dick/moby_dick.txt' file exists.")
        return

    chunks = load_and_chunk_text(DATA_FILE_PATH)
    print(f"Loaded {len(chunks)} text chunks.")

    print("Initializing vector store...")
    vector_store = VectorStore()

    print("Building and saving FAISS index...")
    vector_store.build_index(chunks)
    vector_store.save_index(INDEX_SAVE_DIR)

    print("\nData ingestion complete!")
    print(f"The FAISS index and text chunks are saved in the '{INDEX_SAVE_DIR}' directory.")

if __name__ == "__main__":
    main()
