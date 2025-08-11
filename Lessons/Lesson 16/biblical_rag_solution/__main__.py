"""
Biblical RAG System - Package Entry Point
Allows running the system with: python -m biblical_rag
"""
from dotenv import load_dotenv

from . import launch_gradio_app

if __name__ == "__main__":
    load_dotenv()
    launch_gradio_app()
