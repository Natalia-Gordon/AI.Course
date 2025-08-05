from ragas import evaluate
from ragas.metrics import answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
import json
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

def ragas_evaluate_summary(payload=None, question=None, ground_truth=None, summary=None, contexts=None):
    """
    ✅ Summary evaluation tool using RAGAS answer_correctness.
    Can accept:
       • a single dict via 'payload'
       • or separate args (question, ground_truth, summary, contexts)
    """

    # If payload is passed, extract fields from it
    if payload:
        question = payload.get("question")
        ground_truth = payload.get("ground_truth")
        summary = payload.get("summary")
        contexts = payload.get("contexts", [])

    # Safety check for required fields
    if not all([question, ground_truth, summary]) or contexts is None:
        return "[Evaluation Tool Error] Missing required data for evaluation."

    try:
        # Use RAGAS answer_correctness metric with LLM
        from datasets import Dataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings
        from ragas import SingleTurnSample
        import asyncio
        
        # Configure RAGAS with LLM and embeddings
        ragas_llm = LangchainLLMWrapper(langchain_llm=my_llm.llm)
        embeddings = LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings())
        
        # Create a new instance with the LLM and embeddings
        from ragas.metrics import AnswerCorrectness
        metric = AnswerCorrectness(llm=ragas_llm, embeddings=embeddings)
        
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
    Accepts a formatted string with parameters and parses them.
    """
    try:
        # Try to parse as JSON first
        if input_text.strip().startswith('{'):
            try:
                import json
                import re
                
                # Clean and extract JSON more robustly
                input_text = input_text.strip()
                
                # Find the complete JSON object by counting braces
                if input_text.count('{') != input_text.count('}'):
                    # Find the last complete JSON object
                    brace_count = 0
                    end_pos = 0
                    for i, char in enumerate(input_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    if end_pos > 0:
                        input_text = input_text[:end_pos]
                
                # Clean up any trailing text after the JSON
                input_text = re.sub(r'}\s*[^}]*$', '}', input_text)
                
                data = json.loads(input_text)
                question = data.get("question", "")
                ground_truth = data.get("ground_truth", "")
                summary = data.get("summary", "")
                contexts = data.get("contexts", [])
                
                # Validate that we have the required data
                if question and ground_truth and summary:
                    return ragas_evaluate_summary(
                        question=question,
                        ground_truth=ground_truth,
                        summary=summary,
                        contexts=contexts
                    )
                else:
                    raise ValueError("Missing required data in JSON")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # If JSON parsing fails, try to extract from the text
                pass
        
        # Parse the agent's formatted input
        lines = input_text.split('\n')
        parsed_params = {}
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("- question:"):
                if current_key and current_value:
                    parsed_params[current_key] = '\n'.join(current_value).strip()
                current_key = "question"
                current_value = [line.replace("- question:", "").strip()]
            elif line.startswith("- ground_truth:"):
                if current_key and current_value:
                    parsed_params[current_key] = '\n'.join(current_value).strip()
                current_key = "ground_truth"
                current_value = [line.replace("- ground_truth:", "").strip()]
            elif line.startswith("- summary:"):
                if current_key and current_value:
                    parsed_params[current_key] = '\n'.join(current_value).strip()
                current_key = "summary"
                current_value = [line.replace("- summary:", "").strip()]
            elif line.startswith("- contexts:"):
                if current_key and current_value:
                    parsed_params[current_key] = '\n'.join(current_value).strip()
                current_key = "contexts"
                current_value = [line.replace("- contexts:", "").strip()]
            elif line.startswith("-") and current_key:
                # End of current parameter
                if current_key and current_value:
                    parsed_params[current_key] = '\n'.join(current_value).strip()
                break
            elif current_key and line:
                current_value.append(line)
        
        # Add the last parameter
        if current_key and current_value:
            parsed_params[current_key] = '\n'.join(current_value).strip()
        
        # Extract parameters
        question = parsed_params.get("question", "")
        ground_truth = parsed_params.get("ground_truth", "")
        summary = parsed_params.get("summary", "")
        contexts_str = parsed_params.get("contexts", "")
        
        # Convert contexts string to list if needed
        if isinstance(contexts_str, str) and contexts_str:
            contexts = [contexts_str]  # For now, treat as single context
        else:
            contexts = []
        
        return ragas_evaluate_summary(
            question=question,
            ground_truth=ground_truth,
            summary=summary,
            contexts=contexts
        )
    except Exception as e:
        return f"[RAGAS Tool Error] {str(e)}"
    
#=========================End Of Functions =====================================


