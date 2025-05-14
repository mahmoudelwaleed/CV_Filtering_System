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

    # RAG Chain Definition
    def format_docs_with_metadata(docs):
        formatted_context = []
        for i, doc in enumerate(docs):
            doc_name = doc.metadata.get('doc_name', 'Unknown Document')
            chunk_type = doc.metadata.get('chunk_type', 'Unknown Section')
            doc_string = (
                f"--- Retrieved Chunk {i+1} ---\n"
                f"Source Document: {os.path.basename(doc_name)}\n"
                f"Section Type: {chunk_type}\n"
                f"Content: {doc.page_content}\n"
                f"---------------------------\n"
            )
            formatted_context.append(doc_string)
        return "\n".join(formatted_context)

    template = """    
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
    "- Match candidates whose CV **explicitly mentions the exact role title** (e.g., 'Data Engineer').\n"
    "- Allow for **minor misspellings or slight variations** (e.g., 'Data Enginer', 'Data Engneer').\n"
    "- If **no candidate mentions the role (even approximately)**, return exactly: *'Exact role not mentioned in any CV.'*\n\n"

    "**Output Format:**\n"
    "- Return a list of candidate **names only**, one per line.\n"
    "- Do **not** include summaries, roles, or any other information.\n"
    "- Include **all candidates** who meet the criteria.\n"

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs_with_metadata, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI ---
st.set_page_config(page_title="CV Information Chatbot", layout="wide")
st.title("📄 CV Information Chatbot")
st.caption("Ask questions about the CVs stored in the Qdrant database. This chatbot has no memory.")

# Load the RAG pipeline (cached)
try:
    rag_chain = load_rag_pipeline()
except Exception as e: 
    st.error(f"Failed to initialize the RAG pipeline: {e}")
    st.error("Please check your environment variables and ensure all services (Azure OpenAI, Qdrant) are accessible.")
    st.stop()


# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_question = st.chat_input("Ask a question about the CVs:")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Display bot response
    with st.chat_message("assistant"):
        try:
            # Use st.write_stream for displaying the streaming response
            # The rag_chain.stream() method returns a generator
            response_stream = rag_chain.stream(user_question)
            full_response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

