from langchain_core.tools import Tool
from langchain.agents import initialize_agent
from pathlib import Path
from pydantic import BaseModel
from langchain.tools import StructuredTool

import gradio as gr
# local Imports
import my_llm
import helper_functions
import chatboot_ui
import tool_functions

class RAGASEvalInput(BaseModel):
    question: str
    ground_truth: str
    summary: str
    contexts: list[str]
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
        description="Evaluates summary accuracy using RAGAS. Takes a formatted string with parameters: - question: [question] - ground_truth: [ground_truth] - summary: [summary] - contexts: [contexts]"
    ),
    Tool(
        name="Store Insurance Incidents to Pinecone",
        func=tool_functions.load_documents_to_pinecone,
        description="Upload PDF insurance reports to Pinecone vector DB for semantic search and later retrieval."
    ),
    Tool(
        name="Insurance QnA", #chatbot tool
        func=tool_functions.chat_with_documents,
        description="Ask questions about stored insurance documents in Pinecone vector DB"
    ),
    Tool(
        name="Chatbot UI", 
        func=chatboot_ui.chatbot_interface,
        description="Launches a Gradio chatbot for QnA."
    )
]

# === Create the Agent ===
agent = initialize_agent(
    tools=tools,
    llm=my_llm.llm,
    agent_type="openai-functions", # this is IMPORTANT for StructuredTool
    verbose=True
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
ground_truth = (
    "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. "
    "The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. "
    "Items stolen included electronics, jewelry, and cash totaling $15,700. "
    "The incident was discovered the next day and reported to police and insurance on April 14–15."
)

# Prompt for Agent
''''
prompt = f"""
Please analyze the following PDF: {pdf_path}

1. Use Map-Reduce summarization to summarize the document.
2. Extract the intermediate summaries (Map phase) and treat them as context.
3. Evaluate the accuracy of the summary using RAGAS's Answer Accuracy metric.

To call the evaluation tool, provide a JSON string with the following format:
{{
  "question": "{question}",
  "ground_truth": "{ground_truth}",
  "summary": "<the summary from step 1>",
  "contexts": <list of texts from step 2>
}}

Return the final summary, the extracted contexts, and the evaluation score.
"""
'''
'''
prompt = f"""
Please load the PDF: {pdf_path}
into Pinecone for future similarity-based semantic search and RAG evaluations.
"""
'''

prompt = f"""
Search the policy database and answer:
Question: What is the policy number for the house robbery incident?
"""

'''
prompt = f"""
You are analyzing the PDF: {pdf_path}

Your tasks:
1. Use the 'Summarize with Map-Reduce' tool to summarize it.
2. Use the 'Evaluate Summary with RAGAS' tool with the following parameters:
   - question: "{question}"
   - ground_truth: "{ground_truth}"
   - summary: (the summary from step 1)
   - contexts: (the intermediate map contexts from step 1)

Return summary, map contexts (as bullet list), and the accuracy score.
"""
'''
# Use invoke instead of deprecated run
response = agent.invoke({"input": prompt})

print("\n===== Final Output =====")

# Format the complete response for better readability
formatted_response = {
    'input': response.get('input', ''),
    'output': response.get('output', ''),
    'intermediate_steps': response.get('intermediate_steps', [])
}

# Save agent response to log file as plain text with date and time in filename
helper_functions.save_agent_response(
    formatted_response,
    log_dir="Lessons/Lesson 10/Midlle Course/output",
    base_name="house_robbery_incident"
)

# Print a summary of what was accomplished
print("✅ PDF Analysis Complete!")
print(f"📄 PDF Processed: {pdf_path}")
print(f"📝 Summary Generated: {'Yes' if 'summary' in str(response) else 'No'}")
print(f"📊 Evaluation Attempted: {'Yes' if 'Evaluate Summary with RAGAS' in str(response) else 'No'}")
print(f"💾 Response Saved: output/house_robbery_incident_*.txt")
print("\n" + "="*50)
print("FINAL AGENT RESPONSE:")
print("="*50)
print(response.get('output', 'No output generated'))


