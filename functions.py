import requests

# function for Book API
def getAuthorFromTitle(title):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}"
    response = requests.get(url)
    data = response.json()
    
    items = data.get("items")
    if not items:
        return print("No results found for ",title)
    
    # Take the first result
    book = items[0]["volumeInfo"]
    bookTitle = book.get("title", "Unknown Title")
    authors = book.get("authors", ["Unknown Author"])
    
    return f"'{bookTitle}' is written by {', '.join(authors)}."
