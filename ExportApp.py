import streamlit as st
import requests
import os
import zipfile
import io

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

    if st.sidebar.button("Export Selected Books"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for book in selected_books:
                export_book_pages(book, zip_file)
        zip_buffer.seek(0)
        st.sidebar.success("Export completed.")
        st.sidebar.download_button(
            label="Download Exported Books",
            data=zip_buffer,
            file_name="exported_books.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()