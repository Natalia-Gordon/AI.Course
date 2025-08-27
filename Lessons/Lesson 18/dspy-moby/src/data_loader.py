import os

def load_and_chunk_text(file_path: str) -> list[str]:
    """
    Loads text from a file and splits it into chunks based on paragraphs.

    Args:
        file_path: The path to the text file.

    Returns:
        A list of text chunks (paragraphs).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by paragraphs and remove any empty strings or whitespace-only strings
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    
    return chunks

if __name__ == '__main__':
    # Example usage for testing the script directly
    # Assumes the script is run from the project root or that the path is adjusted accordingly
    data_file_path = os.path.join('data', 'moby_dick', 'moby_dick.txt')
    
    if os.path.exists(data_file_path):
        chunks = load_and_chunk_text(data_file_path)
        print(f"Successfully loaded and chunked the text into {len(chunks)} paragraphs.")
        print("First 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ---")
            print(chunk)
            print()
    else:
        print(f"Error: File not found at {data_file_path}")
        print(f"Current working directory: {os.getcwd()}")
