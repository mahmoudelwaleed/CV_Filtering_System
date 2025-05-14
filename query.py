import streamlit as st
import os
import qdrant_client
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant # Correct import for Langchain Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
if os.environ["LANGCHAIN_TRACING_V2"] == "true":
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Validate essential configurations
required_vars = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    "QDRANT_HOST": QDRANT_HOST,
    "QDRANT_API_KEY": QDRANT_API_KEY,
    "QDRANT_COLLECTION_NAME": QDRANT_COLLECTION_NAME,
}
missing_vars = [var_name for var_name, value in required_vars.items() if not value]
if missing_vars:
    st.error(f"Missing essential environment variables: {', '.join(missing_vars)}. Please check your .env file or environment settings.")
    st.stop() # Stop execution if config is missing

# Embedding Model
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

# Chat Model (LLM) - configured for streaming
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
    streaming=True,
)

# Qdrant Client
qdrant_cloud_client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)

# Qdrant Vector Store
vector_store = Qdrant(
    client=qdrant_cloud_client,
    collection_name=QDRANT_COLLECTION_NAME,
    embeddings=embedding_model,
)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 8})

# RAG-Fusion: Related
template = """You are an expert in rewriting and splitting natural language queries for a CV filtering system that uses vector search to retrieve relevant CV chunks. Your task is to:

1. *Rewrite* the input query to make it clear, specific, and optimized for vector search.
2. *Split* the query into sub-queries if it contains multiple distinct requirements (e.g., conjunctions like "and" or multiple roles/skills). Each sub-query should be concise and focus on a single requirement.

*Input Query*: {question}

*Instructions*:
- For rewriting:
  - Clarify ambiguous terms 
  - Specify roles, skills, or tools explicitly .
  - Use professional language typical of CVs.
- For splitting:
  - Identify conjunctions (e.g., "and") or distinct requirements (e.g., multiple roles or skills).
  - Create sub-queries that are independent and searchable (e.g., split "data engineer and Python" into "data engineer role" and "Python proficiency").
  - If the query is simple and cannot be split, return it as a single sub-query.
- Output contains only the sub-queries seprated by new lines.
Output:

"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=12):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

question = "i want candidates that have exceprience in puthon and i want candidates that have exceprience in javaa"

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)