import streamlit as st
import requests
import os
import zipfile
import io
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

# Set up the API credentials and base URL
BS_URL = 'https://bookstack-xwdf6yz6ya-uc.a.run.app'
BS_TOKEN_ID = 'PclJ9hhYkaHtRwryjHUDGgYNf7BG3Bca'
BS_TOKEN_SECRET = 'JxlPrHruZuX0K4sEOOoO7pDMp57nemvP'
EXPORT_DIR = 'data'

def fetch_shelves():
    headers = {
        'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
    }
    try:
        shelves_response = requests.get(f"{BS_URL}/api/shelves", headers=headers)
        shelves_response.raise_for_status()
        return shelves_response.json()['data']
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving shelves: {e}")
        return []

def fetch_books(shelf_id):
    headers = {
        'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
    }
    try:
        shelf_detail_response = requests.get(f"{BS_URL}/api/shelves/{shelf_id}", headers=headers)
        shelf_detail_response.raise_for_status()
        shelf_detail = shelf_detail_response.json()
        return shelf_detail.get('books', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving books for shelf ID {shelf_id}: {e}")
        return []

def fetch_book_contents(book_id):
    headers = {
        'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
    }
    try:
        book_detail_response = requests.get(f"{BS_URL}/api/books/{book_id}", headers=headers)
        book_detail_response.raise_for_status()
        return book_detail_response.json().get('contents', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving contents for book ID {book_id}: {e}")
        return []

def export_book_pages(book, zip_buffer):
    headers = {
        'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
    }
    book_id = book['id']
    book_name = book['name']
    book_slug = book['slug']
    book_dir = f"book_{book_id}_{book_slug}"

    contents = fetch_book_contents(book_id)
    for content in contents:
        if content['type'] == 'chapter':
            chapter_id = content['id']
            chapter_name = content['name']
            chapter_slug = content['slug']
            chapter_dir = f"{book_dir}/chapter_{chapter_id}_{chapter_slug}"

            for page in content['pages']:
                page_id = page['id']
                page_name = page['name']
                page_slug = page['slug']
                try:
                    page_md_response = requests.get(f"{BS_URL}/api/pages/{page_id}/export/markdown", headers=headers)
                    page_md_response.raise_for_status()
                    page_md_content = page_md_response.text

                    page_file_path = f"{chapter_dir}/page_{page_id}_{page_slug}.md"
                    zip_buffer.writestr(page_file_path, page_md_content)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error exporting page {page_name}: {e}")
        elif content['type'] == 'page':
            page_id = content['id']
            page_name = content['name']
            page_slug = content['slug']
            try:
                page_md_response = requests.get(f"{BS_URL}/api/pages/{page_id}/export/markdown", headers=headers)
                page_md_response.raise_for_status()
                page_md_content = page_md_response.text

                page_file_path = f"{book_dir}/page_{page_id}_{page_slug}.md"
                zip_buffer.writestr(page_file_path, page_md_content)
            except requests.exceptions.RequestException as e:
                st.error(f"Error exporting page {page_name}: {e}")

def prepare_rag_pipeline(selected_pages):
    # Create a temporary directory to store the selected pages
    temp_dir = os.path.join(EXPORT_DIR, "rag_temp")
    os.makedirs(temp_dir, exist_ok=True)

    headers = {
        'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
    }

    # Save the selected pages to the temporary directory
    for page in selected_pages:
        page_id = page['id']
        page_name = page['name']
        page_slug = page['slug']
        try:
            page_md_response = requests.get(f"{BS_URL}/api/pages/{page_id}/export/markdown", headers=headers)
            page_md_response.raise_for_status()
            page_md_content = page_md_response.text

            page_file_path = os.path.join(temp_dir, f"page_{page_id}_{page_slug}.md")
            with open(page_file_path, 'w', encoding='utf-8') as file:
                file.write(page_md_content)
        except requests.exceptions.RequestException as e:
            st.error(f"Error exporting page {page_name}: {e}")

    # Load and split documents
    loader = DirectoryLoader(path=temp_dir, recursive=True, glob="*.md")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings and create vector store index
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_documents(docs, embeddings)

    # Create a conversational retrieval chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=vector_store.as_retriever()
    )
    return qa_chain

def main():
    st.title("BookStack Export Application")
    st.sidebar.title("Controls")

    # Fetch shelves
    shelves = fetch_shelves()
    if not shelves:
        return

    # Display shelves with checkboxes
    selected_shelves = []
    for shelf in shelves:
        if st.sidebar.checkbox(f"{shelf['name']} (ID: {shelf['id']})", key=f"shelf_{shelf['id']}"):
            selected_shelves.append(shelf)

    if not selected_shelves:
        st.sidebar.write("Select shelves to view books.")
        return

    # Display books for selected shelves
    selected_books = []
    for shelf in selected_shelves:
        st.markdown(f"### Shelf: {shelf['name']} (ID: {shelf['id']})")
        books = fetch_books(shelf['id'])
        for book in books:
            if st.checkbox(f"{book['name']} (ID: {book['id']})", key=f"book_{book['id']}"):
                selected_books.append(book)

    if not selected_books:
        st.write("Select books to export pages.")
        return

    # Display pages for selected books
    selected_pages = []
    for book in selected_books:
        st.markdown(f"#### Book: {book['name']} (ID: {book['id']})")
        contents = fetch_book_contents(book['id'])
        for content in contents:
            if content['type'] == 'chapter':
                chapter_id = content['id']
                chapter_name = content['name']
                chapter_slug = content['slug']
                st.markdown(f"    - **Chapter: {chapter_name}** (ID: {chapter_id})")

                for page in content['pages']:
                    page_id = page['id']
                    page_name = page['name']
                    page_slug = page['slug']
                    if st.checkbox(f"Page: {page_name} (ID: {page_id})", key=f"page_{page_id}"):
                        selected_pages.append(page)
            elif content['type'] == 'page':
                page_id = content['id']
                page_name = content['name']
                page_slug = content['slug']
                if st.checkbox(f"Page: {page_name} (ID: {page_id})", key=f"page_{page_id}"):
                    selected_pages.append(content)

    # Create columns for layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col3:
        if st.button("Export Selected Books and Pages"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for book in selected_books:
                    export_book_pages(book, zip_file)
            zip_buffer.seek(0)
            st.success("Export completed.")
            st.download_button(
                label="Download Exported Books and Pages",
                data=zip_buffer,
                file_name="exported_books_and_pages.zip",
                mime="application/zip"
            )

        if st.button("Prepare for RAG Pipeline"):
            if not selected_pages:
                st.error("No pages selected for RAG pipeline.")
            else:
                qa_chain = prepare_rag_pipeline(selected_pages)
                st.success("Preparation for RAG pipeline completed.")
                st.session_state.qa_chain = qa_chain

    if "qa_chain" in st.session_state:
        st.title("Chat with the BookStack docs, powered by LangChain 💬📘")
        st.info("Ask me a question about the BookStack library!", icon="📘")

        # Initialize the chat messages history if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []
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
                response = st.session_state.qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                st.session_state.chat_history.append((prompt, response['answer']))
                st.write(f"Assistant: {response['answer']}")

if __name__ == "__main__":
    main()
