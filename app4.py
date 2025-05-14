import streamlit as st
import os
import qdrant_client
from dotenv import load_dotenv
from functools import partial

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch # Added RunnableBranch
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

    # LLM for query generation and potentially smaller final answers
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
        streaming=True,
    )

    # LLM for larger final answers
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

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

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

-VERY IMPORTANT: if the user query is not clear or not asking about candidates responce with "NO-RAG"

Output:

"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(rag_fusion_template_text)

    def reciprocal_rank_fusion(results: list[list], formula_k=10, top_n_limit=None):
        fused_scores = {}
        for docs_list in results:
            if not docs_list:
                continue
            for rank, doc in enumerate(docs_list):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + formula_k)

        if not fused_scores:
            return []
        reranked_results_with_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        if top_n_limit is not None:
            final_docs = [loads(doc_str) for doc_str, score in reranked_results_with_scores[:top_n_limit]]
        else:
            final_docs = [loads(doc_str) for doc_str, score in reranked_results_with_scores]
        return final_docs

    configured_rrf = partial(reciprocal_rank_fusion, formula_k=10, top_n_limit=12)

    def format_docs_with_metadata(docs_from_rrf):
        if not docs_from_rrf:
            return "No relevant documents found after fusion."
        docs_to_format = docs_from_rrf
        formatted_context_list = []
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


    "5. **Multiple Criteria Queries:**\n"
    "- If the question asks for multiple different roles or criteria (e.g., \"candidates for Role A and candidates for Role B\"), return **separate tables** for each.\n"
    "- Clearly label each section/table according to its criteria.\n\n"

    "6. special instrudction :\n" # Note: "instrudction" -> "instruction", "arole" -> "a role"
    "if the user ask for best fit for arole return the candidates with the most relevant experience and skills for that role, even if they don't meet all the criteria.\n\n"

    "**Output format (per group):**\n"
    "### Candidates for [specific criteria]\n"
        
    "Output format:\n"
    "| Candidate ID | Name  | \n" 
    "|--------------|-------|\n"
    "| One          | ...   | \n\n" 

    "Or, if no matches:\n"
    "No candidates match the criteria.\n\n"

    "Context:\n"
    "{context}\n\n"

    "Question: {question}\n\n"

    "Answer:"""
    prompt = ChatPromptTemplate.from_template(main_rag_template_text)

    def invoke_with_model(input_dict): # expects {"context": ..., "question": ..., "token_count": ...}
        selected_model = llm if input_dict["token_count"] <= 16000 else llm4o
        chain_for_final_answer = (
            prompt
            | selected_model
            | StrOutputParser()
        )
        return chain_for_final_answer.invoke({"context": input_dict["context"], "question": input_dict["question"]})

    # --- Build the RAG chain with conditional NO-RAG logic ---

    # Part 1: Chain for generating raw LLM output for queries.
    # Input: {"question": user_question_string}
    # Output: Raw string from LLM (could be sub-queries or "NO-RAG")
    generate_queries_raw_llm_output_chain = (
        prompt_rag_fusion
        | llm # LLM for query generation
        | StrOutputParser()
    )

    # Part 2: The "else" branch of the RAG pipeline (full RAG processing).
    # This sub-chain is invoked if query generation did NOT result in "NO-RAG".
    # Input: A dictionary like {"question": original_question, "raw_queries_output": "multi-line_query_string"}
    # Output: The final answer string from the main RAG LLM.
    rag_processing_sub_chain = (
        RunnablePassthrough.assign(
            # Parse the raw_queries_output (which is a string) into a list of query strings
            parsed_queries=RunnableLambda(
                lambda x: [q.strip() for q in x["raw_queries_output"].split("\n") if q.strip()]
            )
        )
        # Current dict: {..., "raw_queries_output": ..., "parsed_queries": list_of_query_strings}
        | RunnablePassthrough.assign(
            # Generate context using the parsed queries.
            # The lambda x: x["parsed_queries"] extracts the list of queries for the retrieval part.
            context=(
                RunnableLambda(lambda x: x["parsed_queries"]) # Input: list_of_query_strings
                | retriever.map()                             # Output: list of lists of Document objects
                | configured_rrf                              # Output: list of Document objects (fused and reranked)
                | RunnableLambda(format_docs_with_metadata)   # Output: formatted context string
            )
        )
        # Current dict: {..., "parsed_queries": ..., "context": "formatted_context_string"}
        | RunnableLambda(
            # Prepare the dictionary for the invoke_with_model function
            lambda x: {
                "context": x["context"],
                "question": x["question"], # The original question, passed through the chain
                "token_count": get_encoding_length(x["context"])
            }
        )
        # Output of this lambda is the dict required by invoke_with_model
        | RunnableLambda(invoke_with_model) # Invokes the final LLM call, returns answer string
    )

    # Part 3: The main RAG chain definition using RunnableBranch.
    # The entire chain expects an input dictionary: {"question": user_question_string}
    rag_chain = (
        # Step 1: Get the raw output from the query generation LLM and add it to the current dictionary.
        # The generate_queries_raw_llm_output_chain is invoked with the input dict {"question": ...}.
        RunnablePassthrough.assign(
            raw_queries_output=generate_queries_raw_llm_output_chain
        )
        # Output of this step: {"question": ..., "raw_queries_output": "string_from_query_llm"}
        | RunnableBranch(
            # Condition: Check if the raw_queries_output (after stripping and uppercasing) is "NO-RAG".
            # The lambda x: ... is the condition checker. It receives the dict from the previous step.
            (
                lambda x: x["raw_queries_output"].strip().upper() == "NO-RAG",
                # If true (query is "NO-RAG"):
                # This RunnableLambda simply outputs the string "NO-RAG".
                # It receives the dict from the previous step but ignores it to return a fixed string.
                RunnableLambda(lambda x: "NO-RAG")
            ),
            # Else (default) branch: If the condition is false, execute the full RAG processing sub-chain.
            # rag_processing_sub_chain is invoked with the dict from the step before the RunnableBranch.
            rag_processing_sub_chain
        )
    )
    return rag_chain

# --- Streamlit UI ---
# (The Streamlit UI code remains the same as in your original script)
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
            # The rag_chain now expects a dictionary like {"question": user_question}
            response_stream = rag_chain.stream({"question": user_question})
            full_response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})