from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# === Helper functions ====
def save_agent_response(response, log_dir, base_name):
    """Save agent response to a log file with date and time in the filename."""
    import os
    import datetime
    import json
    
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{base_name}_{now_str}.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("PDF ANALYSIS AND EVALUATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write input prompt
        if 'input' in response:
            f.write("INPUT PROMPT:\n")
            f.write("-" * 30 + "\n")
            f.write(response['input'] + "\n\n")
        
        # Write final output
        if 'output' in response:
            f.write("FINAL OUTPUT:\n")
            f.write("-" * 30 + "\n")
            f.write(response['output'] + "\n\n")
        
        # Write intermediate steps if available
        if 'intermediate_steps' in response and response['intermediate_steps']:
            f.write("INTERMEDIATE STEPS:\n")
            f.write("-" * 30 + "\n")
            for i, step in enumerate(response['intermediate_steps'], 1):
                f.write(f"Step {i}:\n")
                f.write(f"  Action: {step[0]}\n")
                f.write(f"  Input: {step[1]}\n")
                f.write(f"  Output: {step[2]}\n")
                f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")
    
    print(f"✅ Agent response saved to: {log_path}")

# === PDF LOADER FUNCTION ===
def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path.strip().strip('"'))  # clean extra chars
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")
    loader = PyPDFLoader(str(path))
    return loader.load()

# === TEXT SPLITTER ===
def split_docs(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

