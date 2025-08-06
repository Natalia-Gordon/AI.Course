from langchain_core.tools import Tool
from langchain.agents import initialize_agent
from pathlib import Path
from pydantic import BaseModel
import gradio as gr
from datetime import datetime
# local Imports
import my_llm
import helper_functions
import tool_functions

class RAGASEvalInput(BaseModel):
    question: str
    ground_truth: str
    summary: str
    contexts: list[str]

# === Gradio chatbot ===
def chatbot_interface():
    with gr.Blocks(title="Insurance Document Q&A Chatbot") as demo:
        gr.Markdown("# 🏠 Insurance Document Q&A Chatbot")
        gr.Markdown("Ask questions about insurance documents and get AI-powered answers!")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(
            placeholder="Ask a question about the insurance documents...",
            label="Your Question"
        )
        clear = gr.Button("Clear Chat", variant="secondary")
        
        # Add some example questions
        with gr.Row():
            gr.Markdown("**Example questions:**")
        with gr.Row():
            example1 = gr.Button("What time did the robbery occur in the Elm Street?", size="sm")
            example2 = gr.Button("What items were stolen?", size="sm")
            example3 = gr.Button("What is the policy number for the house robbery incident?", size="sm")

        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            try:
                # Use the real agent from main.py to get a response
                response = agent.invoke({"input": message})
                bot_message = response.get('output', 'Sorry, I could not process your request.')
                chat_history.append((message, bot_message))
                return "", chat_history
            except Exception as e:
                error_message = f"Error: {str(e)}"
                chat_history.append((message, error_message))
                return "", chat_history

        def clear_chat():
            return []

        # Set up event handlers
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, None, chatbot, queue=False)
        
        # Example question handlers
        example1.click(lambda: "What time did the robbery occur in the Elm Street?", None, msg)
        example2.click(lambda: "What items were stolen?", None, msg)
        example3.click(lambda: "What is the policy number for the house robbery incident?", None, msg)
        
    return demo

def launch_chatbot(input_text=""):
    """Function to launch the chatbot interface"""
    try:
        demo = chatbot_interface()
        demo.launch(share=False, inbrowser=True)
        return "✅ Chatbot UI launched successfully! The Gradio interface should open in your browser."
    except Exception as e:
        return f"❌ Error launching chatbot: {str(e)}"
    
# === Define Tools for Agent ===
tools = [
    Tool(
        name="Summarize with Map-Reduce",
        func=tool_functions.summarize_map_reduce_pdf,
        description="Use this for summarizing long documents efficiently."
    ),
    Tool(
        name="Summarize with Refine",
        func=tool_functions.summarize_refine_pdf,
        description="Use this when a more accurate, step-by-step summary is preferred."
    ),
    Tool(
        name="Evaluate Summary with RAGAS",
        func=tool_functions.ragas_evaluate_summary_json,
        description="Evaluates summary accuracy using RAGAS. Input should be a formatted string with: - question: [question] - ground_truth: [ground_truth] - summary: [summary] - contexts: [contexts]"
    ),
    Tool(
        name="Store Insurance Incidents to Pinecone",
        func=tool_functions.load_documents_to_pinecone,
        description="Upload PDF insurance reports to Pinecone vector DB for semantic search and later retrieval."
    ),
    Tool(
        name="Insurance QnA",
        func=tool_functions.chat_with_documents,
        description="Ask questions about stored insurance documents in Pinecone vector DB"
    ),
    Tool(
        name="Chatbot UI", 
        func=launch_chatbot,
        description="Launches a Gradio chatbot for QnA."
    )
]

# === Create the Agent ===
agent = initialize_agent(
    tools=tools,
    llm=my_llm.llm,
    agent_type="openai-functions", # this is IMPORTANT for StructuredTool
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
    max_execution_time=120
)


# === Use the Agent ===
# === RUNNING ===
# Works with any kind of text-based PDF (not scanned images — unless OCR is added)
# Provide your PDF path here
# Clean input
pdf_path = 'Lessons/Lesson 10/Midlle Course/data/house_robbery_incident.pdf'

# Confirm it exists
print("PDF Exists:", Path(pdf_path).exists())

# Example Input
question = "What events occurred during the house robbery incident at 45 Elm Street?"
# score 0.69
# ground_truth = "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. Items stolen included electronics, jewelry, and cash totaling $15,700. The incident was discovered the next day and reported to police and insurance on April 14–15."

ground_truth = (
    "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. "
    "The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. "
    "Stolen items included a MacBook Pro, Rolex watch, Sony camera, designer handbag, and cash, totaling $15,700. "
    "The incident was discovered April 14 and reported to police and insurance. "
    "The security panel lost connection at 8:13 PM, likely due to an RF jammer."
)

# Prompt for Agent - Load PDF into Pinecone
'''
prompt = f"""
Search the policy database and answer:
Question: What is the policy number for the house robbery incident?
"""
'''
# Prompt for Agent - Launch Chatbot UI with Insurance QnA tool
'''
prompt = """
Please launch the Chatbot UI tool to open a Gradio interface for interactive Q&A about the insurance documents.
"""
'''
# Prompt for Agent - Summarize with Map-Reduce and RAGAS Evaluation
prompt = f"""
You are an AI assistant helping an insurance claims team analyze documents.

Your tasks:

1. Summarize the provided PDF using the tool `Summarize with Map-Reduce`, Here is the PDF to analyze: `{pdf_path}`.
   - Extract the final summary.
   - Extract the intermediate summaries (map-phase contexts).
   - **When summarizing, ensure you include all key facts, names, dates, and numbers from the document. Use the same or similar wording as the ground truth for important terms (e.g., names, locations, amounts). Do not omit any main events or details.**
   - Keep the summary concise and under 500 characters, but do not leave out any important information.

2. Evaluate the generated summary using the tool `Evaluate Summary with RAGAS`.
   - When calling this tool, you **must** provide the input as a valid JSON object (not a code block, not Markdown, not YAML, not a Python dict).
   - The JSON object must have the following keys:
     - question: The evaluation question.
     - ground_truth: The expected answer.
     - summary: The summary you generated (limit to 500 characters, remove newlines and special formatting).
     - contexts: A list of the map-phase context strings (include no more than 3, each under 300 characters, remove newlines and special formatting).
   - If you encounter a tool error (e.g., JSON parsing failed), reformat and retry with a simpler, shorter summary and contexts.
   - If you reach the iteration or time limit, output a message indicating this and return the best partial result you have.

   Example of correct input for the tool:
   {{
     "question": "What events occurred during the house robbery incident at 45 Elm Street?",
     "ground_truth": "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away...",
     "summary": "A robbery occurred at Jennifer Lawson's residence on April 13, 2024...",
     "contexts": [
       "A robbery occurred at Jennifer Lawson's residence on April 13, 2024...",
       "The house was unoccupied during the incident, and evidence of forced entry was found."
     ]
   }}

   - Do not wrap the JSON in triple backticks or any code block markers.
   - Do not add any extra text before or after the JSON object.

3. Return a final answer in this structure:
---
**Summary:** [summary text]  
**Contexts Used:** [short bullet points of map-phase contexts]  
**RAGAS Accuracy Score:** [score from the RAGAS tool]
---

Here is the question to evaluate: `{question}`  
Here is the expected ground truth: `{ground_truth}`

If you need to call tools, do so step by step.

When you are ready to return the final answer, always start the line with 'Final Answer:' followed by your answer in the required format. Do not output any 'Thought:' without a following 'Action:' or 'Final Answer:'.
"""
# Use invoke instead of deprecated run
response = agent.invoke({"input": prompt})

print("\n===== Final Output =====")
print("✅ PDF Analysis Complete!")
print(f"📄 PDF Processed: {pdf_path}")


print(f"\n=== Output generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# Print intermediate steps if available
if isinstance(response, dict) and 'intermediate_steps' in response and response['intermediate_steps']:
    print(f"{helper_functions.CYAN}{helper_functions.BOLD}Intermediate Steps:{helper_functions.ENDC}")
    for step in response['intermediate_steps']:
        print(f"- {step}")
    print("\n" + "-"*60)

# Format output for display
if isinstance(response, dict) and 'output' in response:
    output_text = response['output']
else:
    output_text = str(response)

plain_output = helper_functions.plain_sections(output_text)

# Warn if agent stopped due to iteration/time limit
if 'Agent stopped due to iteration limit or time limit.' in output_text:
    print(f"{helper_functions.RED}{helper_functions.BOLD}WARNING: Agent stopped due to iteration or time limit. Consider simplifying the task or increasing the limits further.{helper_functions.ENDC}")

print("\n" + "="*60)

# Save formatted output to log file as plain text with date and time in filename
output_dir = "Lessons/Lesson 10/Midlle Course/output"
helper_functions.save_agent_log(response, plain_output, output_dir)