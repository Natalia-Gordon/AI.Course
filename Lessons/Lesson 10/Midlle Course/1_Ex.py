from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain.agents import initialize_agent
from pathlib import Path
from langchain.prompts import PromptTemplate
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import os


# Load environment variables from .env file
load_dotenv()

# Configure Pinecone
# Set your Pinecone API key and environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD")  # Or your preferred cloud
PINECONE_METRIC = os.environ.get("PINECONE_METRIC")  # Or "cosine", depending on your use case
INDEX_NAME = "ai-course-index"
DIMENSION = 1536  # Set this to your embedding dimension

# Set your Pinecone API key and environment
pc = Pinecone(api_key=PINECONE_API_KEY)
# Check if index exists, create if not
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

index = pc.Index(INDEX_NAME)
embedding = OpenAIEmbeddings()  # Use the new class from langchain_openai

def upload_docs_to_pinecone(docs, namespace="default"):
    # 1. Get text and metadata from docs
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    # 2. Get embeddings (batched)
    vectors = embedding.embed_documents(texts)  # shape: (len(texts), DIMENSION)

    # 3. Prepare Pinecone upsert payload
    # Each vector must have a unique ID
    items = []
    for i, (vec, meta) in enumerate(zip(vectors, metadatas)):
        items.append({
            "id": f"doc-{i}",
            "values": vec,
            "metadata": meta
        })

    # 4. Upsert to Pinecone
    pc.Index(INDEX_NAME).upsert(
        vectors=items,
        namespace=namespace
    )    

def store_pdf_to_pinecone(file_path: str, namespace="default"):
    docs = load_pdf(file_path)  # your existing loader
    split = split_docs(docs)
    upload_docs_to_pinecone(split, namespace)
    return f"{len(split)} documents uploaded to Pinecone index '{INDEX_NAME}' in namespace '{namespace}'."

# ============== End Configure Pinecone ===================

# Create the LLM Chain - Option A: map_reduce Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === MAP-REDUCE PROMPTS ===
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following document section. Focus on key events, decisions, and include any available time-based details.

Text:
{text}

Summary with timeline:
"""
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Combine the following summaries into a cohesive document-level summary.
Ensure all chronological events are preserved and logically ordered.

Summaries:
{text}

Combined summary with timeline:
"""
)

# === REFINE PROMPTS ===

refine_initial_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Write an initial summary of the following document section.
Highlight key points and extract any time-based or chronological events if available.

Text:
{text}

Initial summary with timeline:
"""
)

refine_update_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""
The current summary is:
----------------
{existing_answer}
----------------

Given the following new section of the document:
----------------
{text}
----------------

Update the summary by:
- Adding important new information
- Including additional events in chronological order
- Keeping the format clear and ordered

Refined summary with updated timeline:
"""
)

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

# === SUMMARIZATION TOOL FUNCTIONS ===
def summarize_refine_pdf(file_path: str) -> str:
    docs = load_pdf(file_path)
    split = split_docs(docs)
    #chain = load_summarize_chain(llm, chain_type="refine")
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=refine_initial_prompt,
        refine_prompt=refine_update_prompt,
        verbose=True
    )
    return chain.run(split)

def summarize_map_reduce_pdf(file_path: str) -> str:
    docs = load_pdf(file_path)
    split = split_docs(docs)
    print(f"Generated {len(split)} documents.")
    #chain = load_summarize_chain(llm, chain_type="map_reduce")
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True,
        return_intermediate_steps=True
    )
    #return chain.run(split)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
    result = chain.invoke({"input_documents": split})

    return {
        "summary": result["output_text"],
        "contexts": result["intermediate_steps"]
    }


def evaluate_summary_accuracy(question: str, ground_truth: str, summary: str, contexts: list[str]) -> str:

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [summary],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    })
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
    )
    return f"Answer Accuracy Score: {result['answer_relevancy']:.2f}"

def evaluate_summary_accuracy_tool(input_text: str) -> str:
    import json

    try:
        data = json.loads(input_text)
        question = data["question"]
        ground_truth = data["ground_truth"]
        summary = data["summary"]
        contexts = data["contexts"]
        return evaluate_summary_accuracy(question, ground_truth, summary, contexts)
    except Exception as e:
        return f"[Evaluation Tool Error] {str(e)}"

# === Define Tools for Agent ===

tools = [
    Tool(
        name="Summarize with Map-Reduce",
        func=summarize_map_reduce_pdf,
        description="Use this for summarizing long documents efficiently."
    ),
    Tool(
        name="Summarize with Refine",
        func=summarize_refine_pdf,
        description="Use this when a more accurate, step-by-step summary is preferred."
    ),
    Tool(
        name="EvaluateAnswerAccuracy",
        description="Evaluates the answer accuracy of generated answers against ground truth using RAGAS.",
        func=evaluate_summary_accuracy_tool
    ), 
    Tool(
        name="Store Insurance Incidents to Pinecone",
        func=store_pdf_to_pinecone,
        description="Upload PDF insurance reports to Pinecone vector DB for semantic search and later retrieval."
    )
]

# === Create the Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    #agent="zero-shot-react-description",
    agent_type="openai-functions",
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
pdf_path = "Lessons/Lesson 10/Midlle Course/data/house_robbery_incident.pdf"
question = "What events occurred during the house robbery incident at 45 Elm Street?"
ground_truth = (
    "The robbery occurred on April 13, 2024, at 45 Elm Street while the homeowner was away. "
    "The intruder entered between 8:00–10:00 PM by disabling the alarm and forcing the backdoor. "
    "Items stolen included electronics, jewelry, and cash totaling $15,700. "
    "The incident was discovered the next day and reported to police and insurance on April 14–15."
)

# Prompt for Agent
'''
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

prompt = f"""
Please load the PDF: {pdf_path}
into Pinecone for future similarity-based semantic search and RAG evaluations.
"""

# Use invoke instead of deprecated run
response = agent.invoke({"input": prompt})

print("\n===== Final Output =====")
print(response)
# Save agent response to log file as plain text with date and time in filename
save_agent_response(
    response,
    log_dir="Lessons/Lesson 10/Midlle Course/output",
    base_name="house_robbery_incident"
)


