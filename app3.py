import streamlit as st
import os
import qdrant_client
from dotenv import load_dotenv
from functools import partial # Import partial

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough ,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
import tiktoken

# --- Load Environment Variables ---
load_dotenv()
encoding = tiktoken.encoding_for_model("gpt-4o")

def get_encoding_length(text):
    """Get the number of tokens in a text string."""
    return len(encoding.encode(text))


@st.cache_resource # Cache the resource to avoid reloading on every interaction
def load_rag_pipeline():
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
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2")

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
        st.stop()

    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )

    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
        streaming=True,
    )

    llm4o = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
        streaming=True,
    )

    qdrant_cloud_client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )

    vector_store = Qdrant(
        client=qdrant_cloud_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # Each sub-query will retrieve this many documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # RAG-Fusion: Query Generation
    rag_fusion_template_text = """You are an expert in rewriting and splitting natural language queries for a CV filtering system that uses vector search to retrieve relevant CV chunks. Your task is to:

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
- Output contains only the sub-queries.
Output:

"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(rag_fusion_template_text)

    generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()]) # Ensure no empty queries
    )

    # RAG-Fusion: Reciprocal Rank Fusion
    # Renamed 'k' to 'formula_k' for clarity, and added 'top_n_limit'
    def reciprocal_rank_fusion(results: list[list], formula_k=10, top_n_limit=None):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents,
            a formula_k used in the RRF formula, and an optional top_n_limit to restrict
            the number of returned documents.
        """
        fused_scores = {}
        for docs_list in results: # docs_list is the list of documents from one sub-query
            if not docs_list: # Handle cases where a sub-query might return no documents
                continue
            for rank, doc in enumerate(docs_list):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + formula_k) # rank is 0-indexed

        if not fused_scores: # No documents were processed
            return []

        reranked_results_with_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        if top_n_limit is not None:
            final_docs = [loads(doc_str) for doc_str, score in reranked_results_with_scores[:top_n_limit]]
        else:
            final_docs = [loads(doc_str) for doc_str, score in reranked_results_with_scores]
        return final_docs

    configured_rrf = partial(reciprocal_rank_fusion, formula_k=10, top_n_limit=12)

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | configured_rrf


    # This is the correct and complete definition of format_docs_with_metadata
    def format_docs_with_metadata(docs_from_rrf):
        if not docs_from_rrf:
            return "No relevant documents found after fusion."

        docs_to_format = docs_from_rrf

        formatted_context_list = [] # Initialize a list to hold formatted document strings
        for i, doc in enumerate(docs_to_format):
            if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                continue 

            doc_name = doc.metadata.get('doc_name', 'Unknown Document')
            chunk_type = doc.metadata.get('chunk_type', 'Unknown Section')
            doc_string = (
                f"--- Retrieved Chunk {i+1} ---\n"
                f"Source Document: {os.path.basename(doc_name)}\n"
                f"Section Type: {chunk_type}\n"
                f"Content: {doc.page_content}\n"
                f"---------------------------\n"
            )
            formatted_context_list.append(doc_string)
        # Join the list into a single string or return the message if the list is empty
        return "\n".join(formatted_context_list) if formatted_context_list else "No relevant documents found after fusion and formatting."

    main_rag_template_text ="""
    "You are a recruitment analyst reviewing candidate CVs.\n\n"

    "**Instructions:**\n"
    "1. **Thorough Review:**\n"
    "- Carefully review each CV **line by line** to ensure no relevant information is missed.\n"
    "- Pay close attention to job titles, experience, and specific skills.\n\n"

    "2. **Match All Conditions (AND logic):**\n"
    "- Only include candidates who meet **every specified requirement**.\n"
    "- If **no candidate meets all conditions**, return exactly: *'No candidates meet all the specified criteria.'*\n\n"

    "3. **Years of Experience:**\n"
    "- Interpret 'X years of experience' as **'X or more years'**.\n"
    "- Count overlapping or combined relevant experience if applicable.\n\n"

        "4. **Exact Role Matching with Flexibility:**\n"
    "- Match candidates whose CV **explicitly mentions the exact role title** (e.g., 'Data Engineer') uness exceplicitly asked by the user to return best.\n"
    "- Allow for **minor misspellings or slight variations** (e.g., 'Data Enginer', 'Data Engneer').\n"
    "- If **no candidate mentions the role (even approximately)**, return exactly: *'Exact role not mentioned in any CV.'*\n\n"


    5. **Multiple Criteria Queries:**
    - If the question asks for multiple different roles or criteria (e.g., "candidates for Role A and candidates for Role B"), return **separate tables** for each.
    - Clearly label each section/table according to its criteria.

    6. special instrudction :
    if the user ask for best fit for arole return the candidates with the most relevant experience and skills for that role, even if they don't meet all the criteria.

    **Output format (per group):**
    ### Candidates for [specific criteria]
        
Output format:
| Candidate ID | Name  | 
|--------------|-------|
| One          | ...   | 

Or, if no matches:
No candidates match the criteria.

    Context:
    {context}

    Question: {question}

    Answer:"""


    prompt = ChatPromptTemplate.from_template(main_rag_template_text)

    # Define a function to invoke the chain with the selected model
    def invoke_with_model(input_dict):
        selected_model = llm if input_dict["token_count"] <= 16000 else llm4o 
        chain = (
            prompt
            | selected_model
            | StrOutputParser()
        )
        return chain.invoke({"context": input_dict["context"], "question": input_dict["question"]})

    rag_chain = (
        {"context": retrieval_chain_rag_fusion | format_docs_with_metadata, "question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {**x, "token_count": get_encoding_length(x["context"])})
        | RunnableLambda(invoke_with_model)
    )
    return rag_chain




# --- Streamlit UI ---
st.set_page_config(page_title="CV Information Chatbot", layout="wide")
st.title("📄 Hire Match")
st.caption("Ask questions about the CVs stored in the Qdrant database. This chatbot has no memory and uses RAG-Fusion for enhanced retrieval.")

try:
    rag_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Failed to initialize the RAG pipeline: {e}")
    st.error("Please check your environment variables and ensure all services (Azure OpenAI, Qdrant) are accessible.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about the CVs:")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            # The rag_chain expects a dictionary with a "question" key for the RunnablePassthrough for "question",
            # and this same input is also passed to retrieval_chain_rag_fusion (which has generate_queries taking {question}).
            response_stream = rag_chain.stream({"question": user_question}) # Pass question directly
            full_response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})