import gradio as gr
from langchain.schema.runnable import RunnableLambda


def start_gradio(chain: RunnableLambda, starting_message: str):
    def chat_stream(message, history):
        ## This is a generator function, where each call will yield the next entry
        buffer = ""
        for token in chain.stream({"input": message}):
            buffer += token
            yield buffer

    # Set a starting message for the chat
    starting_message = [
        [None, "Hello! I'm a rhyming bot. Ask me anything, and I'll reply in rhyme!"]
    ]

    chatbot_component = gr.Chatbot(starting_message)

    demo = gr.ChatInterface(chat_stream, chatbot=chatbot_component).queue()

    window_kwargs = {}  # or {"server_name": "0.0.0.0", "root_path": "/7860/"}
    demo.launch(share=True, debug=True, **window_kwargs)
