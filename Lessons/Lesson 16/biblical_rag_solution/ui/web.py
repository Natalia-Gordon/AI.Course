"""
Enhanced Gradio interface for Biblical RAG with hybrid retrieval support.
"""
import logging
import os

import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import Config
from ..core.rag_pipeline import RagPipeline
from ..retrievers import build_tfidf_index
from ..retrievers.factory import RetrieverFactory
from ..storage import is_cache_valid, load_from_cache, save_to_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Global components
rag_system = None
retriever_factory = None
config = None

def initialize_rag_system():
    """Initialize the RAG system with hybrid retrieval support."""
    global rag_system, retriever_factory, config
    
    if rag_system is not None:
        return rag_system
    
    try:
        print("🔄 Initializing Biblical RAG system with hybrid retrieval...")
        
        # Initialize configuration
        config = Config()
        
        # Get data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        # Initialize retriever factory
        retriever_factory = RetrieverFactory(config)
        
        # Check cache status
        cache_status = retriever_factory.get_cache_status()
        print(f"� Cache status: TF-IDF={cache_status['tfidf_cache_valid']}, Embeddings={cache_status['embedding_cache_valid']}")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # Initialize RAG pipeline with factory
        rag_system = RagPipeline(llm=llm, retriever_factory=retriever_factory)
        
        # Set default retriever (TF-IDF for backwards compatibility)
        rag_system.set_retriever(config.ui.default_retrieval_method)
        
        print(f"✅ RAG system ready with {config.ui.default_retrieval_method} retrieval!")
        return rag_system
        
    except Exception as e:
        error_msg = f"❌ Error initializing RAG system: {str(e)}"
        print(error_msg)
        raise e

def chat_with_biblical_texts(message, history, retriever_type="hybrid", top_k=5, show_scores=False):
    """
    Enhanced chat function that handles questions about Biblical texts with multiple retrieval options.
    
    Args:
        message (str): User's message/question
        history (List): Chat history
        retriever_type (str): Type of retrieval ("tfidf", "semantic", "hybrid")
        top_k (int): Number of sources to retrieve
        show_scores (bool): Whether to show similarity scores
        
    Returns:
        str: Response from the system
    """
    try:
        # Initialize system if needed
        if rag_system is None:
            initialize_rag_system()
        
        if not message.strip():
            return "Please ask me a question about Biblical texts!"
        
        # Get answer from RAG system with specified retriever
        result = rag_system.answer_question(message, retriever_type=retriever_type, top_k=top_k)
        answer = result.get("answer", "I couldn't generate an answer.")
        sources = result.get("sources", [])
        scores = result.get("scores", [])
        used_retriever = result.get("retriever_type", retriever_type)
        
        # Format response
        response = answer
        
        # Add retriever info
        response += f"\n\n**🔍 Search Method:** {used_retriever.title()}"
        
        if sources:
            response += f"\n\n**📚 Sources ({len(sources)} found):**\n"
            # Ensure scores list has same length as sources (pad with 0.0 if needed)
            scores_padded = scores[:len(sources)] + [0.0] * max(0, len(sources) - len(scores))
            
            for i, (source, score) in enumerate(zip(sources[:top_k], scores_padded[:top_k]), 1):
                book = source.metadata.get('book', 'Unknown')
                chapter = source.metadata.get('chapter', '')
                verse = source.metadata.get('verse', '')
                
                citation = f"{book.title()}"
                if chapter and verse:
                    citation += f" {chapter}:{verse}"
                elif chapter:
                    citation += f" Chapter {chapter}"
                
                score_text = f" (Score: {score:.3f})" if show_scores and score > 0 else ""
                response += f"- {citation}{score_text}\n"
        
        return response
        
    except Exception as e:
        return f"❌ Sorry, I encountered an error: {str(e)}"

def create_gradio_interface():
    """Create an enhanced chat interface with retrieval method selection."""
    global config
    
    # Initialize config if not already done
    if config is None:
        config = Config()
    
    with gr.Blocks(title="Biblical Text Chat - Hybrid Retrieval", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 📚 Biblical Text Chat - Hybrid Retrieval System
        
        Ask me anything about Biblical texts! Choose your preferred search method below.
        
        **Search Methods:**
        - **TF-IDF (Lexical)**: Traditional keyword-based search
        - **Semantic**: AI-powered contextual understanding 
        - **Hybrid**: Combines both methods for best results
        
        **Chunking Strategy:** Balanced approach - Tanach texts grouped by chapters, other sources by complete works for optimal retrieval balance.
        
        **Example questions:**
        - "What does the Bible say about love?"
        - "Tell me about the creation story"
        - "What are the teachings on justice?"
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Get absolute paths for avatar images
                assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
                user_avatar = os.path.join(assets_dir, 'user_avatar.svg')
                bot_avatar = os.path.join(assets_dir, 'bot_avatar.svg')
                
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    show_label=False,
                    avatar_images=(user_avatar, bot_avatar),
                    type='messages'
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Search Settings")
                
                retriever_type = gr.Dropdown(
                    choices=["tfidf", "semantic", "hybrid"],
                    value="hybrid",
                    label="Search Method",
                    info="Choose how to search through Biblical texts"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Sources",
                    info="How many source passages to retrieve"
                )
                
                show_scores = gr.Checkbox(
                    value=False,
                    label="Show Similarity Scores",
                    info="Display relevance scores for each source"
                )
                
                gr.Markdown("### 📊 Quick Actions")
                
                # Show current chunk configuration
                chunk_info = f"**Current Chunking:** {config.chunk.chunk_by}"
                if config.chunk.chunk_by == "multi_verse":
                    chunk_info += f" ({config.chunk.verses_per_chunk} verses per chunk)"
                gr.Markdown(chunk_info)
                
                compare_btn = gr.Button(
                    "🔍 Compare All Methods", 
                    variant="secondary",
                    size="sm"
                )
                
                clear_cache_btn = gr.Button(
                    "🗑️ Clear Cache", 
                    variant="secondary",
                    size="sm"
                )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about Biblical texts...",
                show_label=False,
                scale=4
            )
            submit = gr.Button("Send", variant="primary", scale=1)
        
        gr.Markdown("""
        *Powered by TF-IDF, Semantic Search, and Gemini AI*
        
        **Note:** First-time setup may take a few minutes to build search indices.
        """)
        
        # State for comparison mode
        comparison_mode = gr.State(False)
        
        def respond(message, chat_history, ret_type, k, scores, comp_mode):
            if comp_mode:
                # Comparison mode - show results from all methods
                try:
                    if rag_system is None:
                        initialize_rag_system()
                    
                    results = rag_system.compare_retrievers(message, top_k=k)
                    
                    response = f"**🔍 Comparison Results for:** \"{message}\"\n\n"
                    
                    for method, result in results.items():
                        response += f"### {method.upper()} Method\n"
                        if "error" in result:
                            response += f"❌ Error: {result['error']}\n\n"
                        else:
                            response += f"{result['answer'][:200]}...\n"
                            if result.get('sources'):
                                response += f"*Sources: {len(result['sources'])} found*\n\n"
                    
                    # Use messages format
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history, False  # Reset comparison mode
                    
                except Exception as e:
                    error_response = f"❌ Comparison failed: {str(e)}"
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": error_response})
                    return "", chat_history, False
            else:
                # Normal mode
                bot_message = chat_with_biblical_texts(message, chat_history, ret_type, k, scores)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                return "", chat_history, comp_mode
        
        def enable_comparison():
            return True
        
        def clear_cache():
            try:
                if retriever_factory is not None:
                    retriever_factory.clear_cache()
                    gr.Info("✅ Cache cleared successfully!")
                    return "✅ Cache cleared successfully!"
                else:
                    gr.Warning("⚠️ System not initialized yet")
                    return "⚠️ System not initialized yet"
            except Exception as e:
                error_msg = f"❌ Error clearing cache: {str(e)}"
                gr.Error(error_msg)
                return error_msg
        
        # Event handlers
        msg.submit(
            respond, 
            [msg, chatbot, retriever_type, top_k, show_scores, comparison_mode], 
            [msg, chatbot, comparison_mode]
        )
        submit.click(
            respond, 
            [msg, chatbot, retriever_type, top_k, show_scores, comparison_mode], 
            [msg, chatbot, comparison_mode]
        )
        
        compare_btn.click(
            enable_comparison,
            outputs=[comparison_mode]
        )
        
        clear_cache_btn.click(
            clear_cache
        )
    
    return demo

if __name__ == "__main__":
    # Initialize the system on startup
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"Warning: Could not initialize on startup: {e}")
        print("Will try to initialize when first question is asked.")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
