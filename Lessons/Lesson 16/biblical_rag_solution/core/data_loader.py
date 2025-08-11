import json
import os
import logging
from typing import Iterator, Dict, Any, Tuple

# Configure basic logging to print to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_jsonl_file(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Loads a single JSONL file line by line as a generator, handling potential errors.

    Args:
        file_path (str): The path to the JSONL file.

    Yields:
        Iterator[Dict[str, Any]]: A generator of dictionaries, one for each valid line.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping corrupted JSON line {line_num} in {file_path}")
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {file_path}: {e}")

def load_all_data_generator(data_dir: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Loads all JSONL files from a directory structure, yielding each record
    one by one along with its category and source filename. This approach is memory-efficient.

    Args:
        data_dir (str): The path to the data directory (e.g., 'data/').

    Yields:
        Iterator[Tuple[str, str, Dict[str, Any]]]: A generator of tuples, where each tuple
        contains the category (e.g., 'bavli'), source filename, and the data record.
    """
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return

    print("\n--- Starting Data Loading Process ---")
    for category in sorted(os.listdir(data_dir)):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            print(f"Processing category: {category}")
            files_in_category = sorted([f for f in os.listdir(category_dir) if f.endswith('.jsonl')])
            if not files_in_category:
                logging.warning(f"No .jsonl files found in category: {category}")
                continue
            
            for filename in files_in_category:
                file_path = os.path.join(category_dir, filename)
                # Simple print to show progress, as requested
                print(f"  -> Loading file: {filename}")
                for record in load_jsonl_file(file_path):
                    yield (category, filename, record)
    print("--- Data Loading Process Finished ---\n")

# Example of how to use the generator (for testing purposes)
if __name__ == '__main__':
    # Construct the absolute path to the data directory relative to this script's location
    # __file__ -> src/data_loader.py
    # os.path.dirname(__file__) -> src/
    # os.path.join(..., '..') -> (project root)
    # os.path.join(..., 'data') -> data/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(project_root, 'data')
    
    print(f"Starting data loading example from: {data_directory}")
    
    # The generator is stateful; you can only iterate over it once.
    data_generator = load_all_data_generator(data_directory)
    
    # Process the first 5 records as a sample to demonstrate usage
    print("\n--- Fetching first 5 records as a sample ---")
    for i, (category, source_file, data_record) in enumerate(data_generator):
        if i >= 5:
            break
        print(f"\nRecord {i+1}:")
        print(f"  Category: {category}")
        print(f"  Source File: {source_file}")
        # Print a snippet of the content to keep it brief
        content_snippet = data_record.get('text', '')[:100].replace('\n', ' ')
        # The line below makes the snippet safe to print on Windows by replacing characters that the console can't display.
        content_snippet = content_snippet.encode('cp1252', 'replace').decode('cp1252')
        print(f"  Content (snippet): {content_snippet}...")
        print("-" * 20)

    print("\nData loading example finished successfully.")