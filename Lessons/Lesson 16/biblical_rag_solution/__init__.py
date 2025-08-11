# Biblical RAG package
from .config import Config
from .core import load_all_data_generator
from .core.rag_pipeline import RagPipeline
from .retrievers.factory import RetrieverFactory
from .storage import EmbeddingCacheManager, TFIDFCacheManager

__version__ = "1.0.0"
__all__ = [
    "Config", 
    "RagPipeline", 
    "RetrieverFactory", 
    "load_all_data_generator",
    "TFIDFCacheManager",
    "EmbeddingCacheManager"
]

# Gradio app entry point
def launch_gradio_app():
    """Launch the Gradio web interface."""
    from .ui.web import create_gradio_interface
    interface = create_gradio_interface()
    interface.launch()
