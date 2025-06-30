# EXERCISE: Multi-Style Chatbot with Dynamic System Prompts
#
# Complete the following code to implement a chatbot that responds in different styles
# (Rhyme, Haiku, Shakespearean, Normal) based on user selection. The Gradio UI is provided.
#
# 1. Define a dictionary mapping style names to system prompts.
# 2. Create a function to build a LangChain chain based on the selected style.
# 3. Implement a callback function for Gradio that handles user input and updates the chat history.
# 4. Wire up the Gradio UI to use your callback and launch the app.

from gradio_multi_style_interface import create_multi_style_gradio_ui

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define your STYLES dictionary here
STYLES = {
    "Rhyme": "",
    "Haiku": "",
    ...
}

# Initialize your LLM here
chat_llm = ...

def get_chain(style):
    """
    Given a style, return a LangChain chain that uses the appropriate system prompt.
    """
    pass

def user_respond(message, history, style):
    """
    Handle user input, run the chain, and update the chat history.
    """
    pass

if __name__ == "__main__":
    demo = create_multi_style_gradio_ui(user_respond)
    demo.launch()
