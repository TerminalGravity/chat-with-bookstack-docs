import requests
import os

# Set up the API credentials and base URL
BS_URL = 'https://bookstack-xwdf6yz6ya-uc.a.run.app'
BS_TOKEN_ID = 'PclJ9hhYkaHtRwryjHUDGgYNf7BG3Bca'
BS_TOKEN_SECRET = 'JxlPrHruZuX0K4sEOOoO7pDMp57nemvP'
EXPORT_DIR = 'data'

# Ensure the export directory exists
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# Authenticate with the API (assuming the API uses token-based authentication)
headers = {
    'Authorization': f'Token {BS_TOKEN_ID}:{BS_TOKEN_SECRET}'
}

# Retrieve the list of shelves
shelves_response = requests.get(f"{BS_URL}/api/shelves", headers=headers)
shelves_response.raise_for_status()
shelves = shelves_response.json()['data']

# Print the IDs and names of the shelves found
print("Found shelves:", [(shelf['id'], shelf['name']) for shelf in shelves])

# Iterate over each shelf to get its books
for shelf in shelves:
    shelf_id = shelf['id']
    shelf_name = shelf['name']
    print(f"\nShelf: {shelf_name} (ID: {shelf_id})")

    # Retrieve the list of books in the shelf
    shelf_books_response = requests.get(f"{BS_URL}/api/shelves/{shelf_id}/books", headers=headers)
    shelf_books_response.raise_for_status()
    shelf_books = shelf_books_response.json()['data']

    # Print the IDs and names of the books found in the shelf
    print("  Books:", [(book['id'], book['name']) for book in shelf_books])

    # Iterate over each book to get its chapters and pages
    for book in shelf_books:
        book_id = book['id']
        book_name = book['name']
        book_slug = book['slug']
        print(f"  Book: {book_name} (ID: {book_id})")

        book_dir = os.path.join(EXPORT_DIR, f"book_{book_id}_{book_slug}")
        os.makedirs(book_dir, exist_ok=True)

        # Fetch the details of the book, which includes the chapters and pages
        book_detail_response = requests.get(f"{BS_URL}/api/books/{book_id}", headers=headers)
        book_detail_response.raise_for_status()
        book_contents = book_detail_response.json()

        # Use the 'contents' key instead of 'content'
        if 'contents' in book_contents:
            contents = book_contents['contents']
        else:
            print(f"    Expected 'contents' key not found in response for book ID {book_id}.")
            print("    Response keys:", book_contents.keys())
            continue  # Skip to the next book

        # Save each page's markdown content to a file, organized by chapter
        for content in contents:
            if content['type'] == 'chapter':
                chapter_id = content['id']
                chapter_name = content['name']
                chapter_slug = content['slug']
                print(f"    Chapter: {chapter_name} (ID: {chapter_id})")

                chapter_dir = os.path.join(book_dir, f"chapter_{chapter_id}_{chapter_slug}")
                os.makedirs(chapter_dir, exist_ok=True)

                for page in content['pages']:
                    page_id = page['id']
                    page_name = page['name']
                    page_slug = page['slug']
                    print(f"      Page: {page_name} (ID: {page_id})")

                    page_md_response = requests.get(f"{BS_URL}/api/pages/{page_id}/export/markdown", headers=headers)
                    page_md_response.raise_for_status()
                    page_md_content = page_md_response.text

                    page_file_path = os.path.join(chapter_dir, f"page_{page_id}_{page_slug}.md")
                    with open(page_file_path, 'w', encoding='utf-8') as file:
                        file.write(page_md_content)
            elif content['type'] == 'page':
                # Pages that are not in a chapter are saved directly in the book directory
                page_id = content['id']
                page_name = content['name']
                page_slug = content['slug']
                print(f"    Page: {page_name} (ID: {page_id})")

                page_md_response = requests.get(f"{BS_URL}/api/pages/{page_id}/export/markdown", headers=headers)
                page_md_response.raise_for_status()
                page_md_content = page_md_response.text

                page_file_path = os.path.join(book_dir, f"page_{page_id}_{page_slug}.md")
                with open(page_file_path, 'w', encoding='utf-8') as file:
                    file.write(page_md_content)

print("Export completed.")