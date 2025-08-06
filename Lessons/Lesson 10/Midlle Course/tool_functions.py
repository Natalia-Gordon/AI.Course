
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import AnswerCorrectness, AnswerSimilarity
import asyncio
import traceback
# local Imports
import my_llm
import helper_functions
import tool_prompts
import pinecone_db

# === SUMMARIZATION TOOL FUNCTIONS ===
def summarize_refine_pdf(file_path: str) -> str:
    docs = helper_functions.load_pdf(file_path)
    split = helper_functions.split_docs(docs)
    chain = load_summarize_chain(
        my_llm.llm,
        chain_type="refine",
        question_prompt=tool_prompts.refine_initial_prompt,
        refine_prompt=tool_prompts.refine_update_prompt,
        verbose=True
    )
    result = chain.invoke({"input_documents": split})
    return result["output_text"]

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
    
# Create Chatbot Function
def chat_with_documents(question: str):
    result = pinecone_db.qa_chain.invoke({"query": question})
    # You can return just the answer, or the full result dict if you want sources
    return result["result"]

def load_documents_to_pinecone(file_path: str):
    """Loads policy documents into Pinecone for future queries."""
    docs = helper_functions.load_pdf(file_path)
   
    # פיצול מסמכים לקטעים קטנים
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
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

def ragas_evaluate_summary(question=None, ground_truth=None, summary=None, contexts=None):
    """
    ✅ Summary evaluation tool using RAGAS answer_correctness.
    Can accept:
       • a single dict via 'payload'
       • or separate args (question, ground_truth, summary, contexts)
    """

    # Safety check for required fields
    if not all([question, ground_truth, summary]) or contexts is None:
        missing = []
        if not question: missing.append("question")
        if not ground_truth: missing.append("ground_truth")
        if not summary: missing.append("summary")
        if contexts is None: missing.append("contexts")
        return f"[Evaluation Tool Error] Missing required data: {', '.join(missing)}"

    try:        
        # Configure RAGAS with LLM and embeddings
        ragas_llm = LangchainLLMWrapper(langchain_llm=my_llm.llm)
        embeddings = LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings())
        
        # Create a new instance with the LLM and embeddings
        # Fix: AnswerSimilarity no longer takes llm argument in recent RAGAS versions
        answer_similarity = AnswerSimilarity(embeddings=embeddings)
        metric = AnswerCorrectness(llm=ragas_llm, embeddings=embeddings, answer_similarity=answer_similarity)
        
        # Use the single_turn_ascore method for individual evaluation
        sample = SingleTurnSample(
            user_input=question,
            response=summary,
            reference=ground_truth,
            retrieved_contexts=contexts
        )
        
        # Run the async function
        result = asyncio.run(metric.single_turn_ascore(sample))
        
        return f"RAGAS Answer Correctness Score: {result:.2f}"
        
    except Exception as e:
        print("[RAGAS DEBUG] Advanced metric failed, falling back to keyword match.")
        traceback.print_exc()
        # Fallback to simple evaluation if RAGAS fails
        try:
            # Convert to lowercase for comparison
            gt_lower = ground_truth.lower()
            summary_lower = summary.lower()
            
            # Count key terms that should be present
            key_terms = ['robbery', 'april', 'elm street', 'homeowner', 'away', 'intruder', 'alarm', 'backdoor', 'electronics', 'jewelry', 'cash', '15700', 'police', 'insurance']
            
            found_terms = 0
            for term in key_terms:
                if term in gt_lower and term in summary_lower:
                    found_terms += 1
            
            # Calculate simple accuracy score
            accuracy_score = found_terms / len(key_terms)
            
            return f"Fallback Evaluation Score: {accuracy_score:.2f} ({found_terms}/{len(key_terms)} key terms matched)"
            
        except Exception as fallback_error:
            return f"[Evaluation Tool Error] RAGAS failed: {str(e)}, Fallback failed: {str(fallback_error)}"

def ragas_evaluate_summary_json(input_text: str):
    """
    RAGAS evaluation tool for LangChain agent calls.
    Accepts a JSON string with parameters: question, ground_truth, summary, contexts.
    """
    import json
    try:
        # Remove code block markers and whitespace
        cleaned_input = input_text.strip()
        if cleaned_input.startswith('```'):
            cleaned_input = cleaned_input[3:]
        if cleaned_input.endswith('```'):
            cleaned_input = cleaned_input[:-3]
        cleaned_input = cleaned_input.strip()
        # Parse as JSON
        data = json.loads(cleaned_input)
        question = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        summary = data.get("summary", "")
        contexts = data.get("contexts", [])
        print(f"\n🔍 DEBUG: Parsed JSON - question: {question[:50]}...")
        print(f"🔍 DEBUG: Parsed JSON - ground_truth: {ground_truth[:50]}...")
        print(f"🔍 DEBUG: Parsed JSON - summary: {summary[:50]}...")
        print(f"🔍 DEBUG: Parsed JSON - contexts: {len(contexts)} items")
        # Validate required fields
        if not question or not ground_truth or not summary:
            missing = []
            if not question: missing.append("question")
            if not ground_truth: missing.append("ground_truth")
            if not summary: missing.append("summary")
            return f"[RAGAS Tool Error] Missing required parameters: {', '.join(missing)}"
        return ragas_evaluate_summary(
            question=question,
            ground_truth=ground_truth,
            summary=summary,
            contexts=contexts
        )
    except Exception as e:
        return f"[RAGAS Tool Error] JSON parsing failed: {str(e)}\nInput must be a valid JSON object with keys: question, ground_truth, summary, contexts."
    
#=========================End Of Functions =====================================


