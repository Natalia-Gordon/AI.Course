from ragas import evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
# local Imports
import my_llm
import helper_functions
import tool_prompts
import pinecone_db

# === SUMMARIZATION TOOL FUNCTIONS ===
def summarize_refine_pdf(file_path: str) -> str:
    docs = helper_functions.load_pdf(file_path)
    split = helper_functions.split_docs(docs)
    #chain = load_summarize_chain(llm, chain_type="refine")
    chain = load_summarize_chain(
        my_llm.llm,
        chain_type="refine",
        question_prompt=tool_prompts.refine_initial_prompt,
        refine_prompt=tool_prompts.refine_update_prompt,
        verbose=True
    )
    return chain.run(split)

def summarize_map_reduce_pdf(file_path: str) -> str:
    docs = helper_functions.load_pdf(file_path)
    split = helper_functions.split_docs(docs)
    print(f"Generated {len(split)} documents.")
    chain = load_summarize_chain(
        my_llm.llm,
        chain_type="map_reduce",
        map_prompt=tool_prompts.map_prompt,
        combine_prompt=tool_prompts.combine_prompt,
        verbose=True,
        return_intermediate_steps=True
    )
    #return chain.run(split)
    #chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
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
    llm_wrapper = LangchainLLMWrapper(my_llm.llm)
    embedding_wrapper = LangchainEmbeddingsWrapper(pinecone_db.embedding)

    faithfulness = Faithfulness(llm=llm_wrapper)
    answer_relevancy = ResponseRelevancy(llm=llm_wrapper)
    context_precision = LLMContextPrecisionWithoutReference(embedding=embedding_wrapper)

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
    
# Create Chatbot Function
def chat_with_documents(question: str):
    result = pinecone_db.qa_chain.invoke({"query": question})
    # You can return just the answer, or the full result dict if you want sources
    return result["result"]

def load_documents_to_pinecone(file_path: str):
    """Loads policy documents into Pinecone for future queries."""
    docs = helper_functions.load_pdf(file_path)
   
    # פיצול מסמכים לקטעים קטנים
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectors = []
    for chunk in chunks:
        vector = pinecone_db.embedding.embed_query(chunk.page_content)
        vectors.append({
            "id": f"doc-{hash(chunk.page_content)}",
            "values": vector,
            "metadata": {"text": chunk.page_content}
        })

    # שמירה ל-Pinecone
    pinecone_db.pinecone_index.upsert(vectors)
    return f"{len(vectors)} chunks were loaded into Pinecone for querying."

#=========================End Of Functions =====================================


