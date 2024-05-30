import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

# Streamlit app configuration
st.set_page_config(
    page_title="Chat with the BookStack docs, powered by LangChain",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Load OpenAI API key from Streamlit secrets
try:
    openai.api_key = st.secrets["openai_key"]
except KeyError:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
    st.stop()

st.title("Chat with the BookStack docs, powered by LangChain 💬📘")
st.info("Ask me a question about the ADR BookStack library!", icon="📘")

# Initialize the chat messages history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Caching and loading data using LangChain
@st.experimental_singleton(show_spinner=False)
def load_data():
    try:
        with st.spinner("Loading and indexing the BookStack docs – hang tight! This should take 1-2 minutes."):
            # Load and split documents
            loader = DirectoryLoader(path="./data", recursive=True, glob="*.md")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Generate embeddings and create vector store index
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
            vector_store = FAISS.from_documents(docs, embeddings)

            # Create a conversational retrieval chain
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=openai.api_key,
                temperature=0.5,
                system_message="You are a friendly and helpful customer support representative for the BookStack Python library. "
                               "Your job is to assist users with their questions and provide clear, concise, and accurate information. "
                               "Ensure that your responses are polite, supportive, and easy to understand."
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm, retriever=vector_store.as_retriever()
            )
            return qa_chain
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

qa_chain = load_data()

# Initialize the chat engine if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt for user input and save to chat history
prompt = st.text_input("Your question", key="prompt")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    st.write(f"{message['role'].capitalize()}: {message['content']}")

# If last message is not from assistant, generate a new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        response = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        st.session_state.chat_history.append((prompt, response['answer']))
        st.write(f"Assistant: {response['answer']}")