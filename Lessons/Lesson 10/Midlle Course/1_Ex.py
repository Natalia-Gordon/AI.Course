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
from fpdf import FPDF
from pathlib import Path
import datetime

# Set your OpenAI key in the .env file
load_dotenv()
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
def save_summary_as_pdf(summary_text: str, file_path: str):
    output_path = Path(file_path)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Document Summary with Timeline", ln=True)

    pdf.set_font("Arial", size=12)
    for line in summary_text.strip().split("\n"):
        pdf.multi_cell(0, 10, line.strip())

    pdf.output(str(output_path))
    print(f"✅ PDF saved to: {output_path.resolve()}")

def save_summary_as_markdown(summary_text: str, file_path: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    markdown_content = f"# Summary with Timeline\n\n*Generated on {timestamp}*\n\n{summary_text}"

    output_path = Path(file_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ Markdown file saved to: {output_path.resolve()}")

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
def summarize_map_reduce_pdf(file_path: str) -> str:
    docs = load_pdf(file_path)
    split = split_docs(docs)
    #chain = load_summarize_chain(llm, chain_type="map_reduce")
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True
    )
    return chain.run(split)

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
    )
]

# === Create the Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# === Use the Agent ===
# === RUNNING ===
# Works with any kind of text-based PDF (not scanned images — unless OCR is added)
# Provide your PDF path here
# Clean input
pdf_path = 'Lessons/Lesson 10/Midlle Course/data/incident_report_short_refine.pdf'

# Confirm it exists
print("PDF Exists:", Path(pdf_path).exists())

# prompt = f"""
#Please summarize the content of the PDF located at: {pdf_path}
#Pick the best summarization tool depending on length and needed accuracy.
#"""

# Use in prompt
prompt = f"""
Please summarize the PDF at the following path:
{pdf_path}
Use the best summarization method depending on document length.
"""
summary = agent.run(prompt)

# Save to both formats
save_summary_as_pdf(summary, file_path="Lessons/Lesson 10/Midlle Course/summary/cyber_summary.pdf")
save_summary_as_markdown(summary, file_path="Lessons/Lesson 10/Midlle Course/summary/cyber_summary.md")

