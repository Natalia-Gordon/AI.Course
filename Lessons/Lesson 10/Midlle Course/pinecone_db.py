from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import os
from langchain_openai import OpenAIEmbeddings
# local Imports
import my_llm
import helper_functions

# Configure Pinecone
# Set your Pinecone API key and environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD")  # Or your preferred cloud
PINECONE_METRIC = os.environ.get("PINECONE_METRIC")  # Or "cosine", depending on your use case
INDEX_NAME = "insurance-claims"
DIMENSION = 1536  # Set this to your embedding dimension

# Set your Pinecone API key and environment
pc = Pinecone(api_key=PINECONE_API_KEY)

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

pinecone_index = pc.Index(INDEX_NAME)
embedding = OpenAIEmbeddings()  # Use the new class from langchain_openai

# Set Up Retriever and RAG Chain
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embedding,
    text_key="text",  # default metadata key
    namespace="default"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=my_llm.llm,
    chain_type="map_reduce",  # or "map_reduce" if you want summarization-like behavior
    retriever=retriever,
    return_source_documents=True
)

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
    pinecone_index.upsert(
        vectors=items,
        namespace=namespace
    )    

def store_pdf_to_pinecone(file_path: str, namespace="default"):
    docs = helper_functions.load_pdf(file_path)  # your existing loader
    split = helper_functions.split_docs(docs)
    upload_docs_to_pinecone(split, namespace)
    return f"{len(split)} documents uploaded to Pinecone index '{INDEX_NAME}' in namespace '{namespace}'."

# ============== End Configure Pinecone ===================
