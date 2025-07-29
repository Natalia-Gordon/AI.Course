from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# === Helper functions ====
def save_agent_response(response, log_dir, base_name):
    """Save agent response to a log file with date and time in the filename."""
    import os
    import datetime
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{base_name}_{now_str}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(str(response))
    print(f"Agent response saved to: {log_path}")

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

