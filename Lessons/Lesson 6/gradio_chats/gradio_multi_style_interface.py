import gradio as gr


def create_multi_style_gradio_ui(user_respond_callback):
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Style Chatbot\nSelect a style and chat!")
        style = gr.Dropdown(
            ["Rhyme", "Haiku", "Shakespearean", "Normal"],
            value="Rhyme",
            label="Response Style",
        )
        starting_message = (
            "Hello! I'm  can write in different styles. What would you like to ask me?"
        )
        chatbot = gr.Chatbot([[None, starting_message]])
        msg = gr.Textbox(label="Your message")
        send_btn = gr.Button("Send")

        send_btn.click(user_respond_callback, [msg, chatbot, style], [chatbot, msg])
        msg.submit(user_respond_callback, [msg, chatbot, style], [chatbot, msg])

    return demo
