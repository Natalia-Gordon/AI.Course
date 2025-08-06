from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from datetime import datetime
import re

# Helper for colored output
HEADER = '\033[95m'
BOLD = '\033[1m'
ENDC = '\033[0m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'

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

def save_agent_log(response, plain_output, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"house_robbery_incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(f"{output_dir}/{filename}", "w", encoding="utf-8") as f:
        f.write(f"=== Output generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        # Write intermediate steps if available
        if isinstance(response, dict) and 'intermediate_steps' in response and response['intermediate_steps']:
            f.write("Intermediate Steps:\n")
            for i, step in enumerate(response['intermediate_steps'], 1):
                f.write(f"Step {i}:\n")
                f.write(f"  {step}\n")
            f.write("\n" + "-"*60 + "\n")
        f.write(plain_output)
        f.write("\n" + "="*60 + "\n")

# === PDF LOADER FUNCTION ===
def load_pdf(file_path: str) -> list[Document]:
    # Clean extra chars, backticks, and whitespace
    path_str = file_path.strip().strip('"').strip("'").strip('`')
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] PDF not found: {path.resolve()}")
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")
    loader = PyPDFLoader(str(path))
    return loader.load()

# === TEXT SPLITTER ===
def split_docs(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def highlight_sections(text):
    # Add color and bold to key sections
    text = re.sub(r"\*\*Summary:\*\*", f"{HEADER}{BOLD}Summary:{ENDC}", text)
    text = re.sub(r"\*\*Contexts Used:\*\*", f"{CYAN}{BOLD}Contexts Used:{ENDC}", text)
    text = re.sub(r"\*\*RAGAS Accuracy Score:\*\*", f"{YELLOW}{BOLD}RAGAS Accuracy Score:{ENDC}", text)
    # Divider
    text = re.sub(r"---", f"{GREEN}{'-'*40}{ENDC}", text)
    return text

def plain_sections(text):
    # Remove ANSI and markdown, keep structure for file
    text = re.sub(r"\*\*Summary:\*\*", "Summary:", text)
    text = re.sub(r"\*\*Contexts Used:\*\*", "Contexts Used:", text)
    text = re.sub(r"\*\*RAGAS Accuracy Score:\*\*", "RAGAS Accuracy Score:", text)
    text = re.sub(r"---", "-"*40, text)
    # Remove any ANSI codes just in case
    text = re.sub(r"\033\[[0-9;]*m", "", text)
    return text

