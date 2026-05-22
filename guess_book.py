from nltk.sem import Expression
read_expr = Expression.fromstring

def prepare_game_data(kb):
    books = set()
    authors = {}
    genres = set()
    book_genres = {}

    for expr in kb:
        #print(expr)
        # Only process atomic facts
        if expr.__class__.__name__ != "ApplicationExpression":
            continue

        pred = str(expr.pred)
        args = expr.args

        # Extract genres
        if len(args) == 1 and pred not in ['Author', 'Book', 'Readable'] and not pred.startswith('-'):
            book = str(args[0])
            books.add(book)
            genres.add(pred)
            book_genres[book] = pred

        # Extract authors
        if pred == 'Author' and len(args) == 1:
            author = str(args[0])
            authors.setdefault(author, [])

        # Map author to books
        if pred in ['Wrote', 'AuthorOf'] and len(args) == 2:
            author = str(args[0])
            book = str(args[1])
            authors.setdefault(author, []).append(book)
            books.add(book)
            if book not in book_genres:
                book_genres[book] = 'Unknown'

    # Combine all info into list of dicts
    books_data = []
    for book in books:
        genre = book_genres.get(book, "Unknown")
        author = "Unknown"
        for a, bks in authors.items():
            if book in bks:
                author = a
                break
        books_data.append({"book": book, "genre": genre, "author": author})

    return books_data, genres, authors


def guessing_game(kb):
    books_data, genres, authors = prepare_game_data(kb)
    remaining_books = books_data.copy()

    print("Let's find a book for you! Please answer Yes or No to the questions.")

    # Ask genre questions first
    for genre in genres:
        answer = input(f"Is your book {genre}? (Yes/No) ").strip().lower()
        if answer in ["yes", "y"]:
            remaining_books = [b for b in remaining_books if b["genre"] == genre]
        else:
            remaining_books = [b for b in remaining_books if b["genre"] != genre]

        if len(remaining_books) == 1:
            break
        elif len(remaining_books) == 0:
            print("No matching books found.")
            return

    # Ask author questions if more than 1 book remains
    if len(remaining_books) > 1:
        for author in authors.keys():
            answer = input(f"Is your book written by {author}? (Yes/No) ").strip().lower()
            if answer in ["yes", "y"]:
                remaining_books = [b for b in remaining_books if b["author"] == author]
            else:
                remaining_books = [b for b in remaining_books if b["author"] != author]

            if len(remaining_books) == 1:
                break
            elif len(remaining_books) == 0:
                print("No matching books found.")
                return

    # Result
    if remaining_books:
        print(f"I think your book is: {remaining_books[0]['book']} (Genre: {remaining_books[0]['genre']}, Author: {remaining_books[0]['author']})")
    else:
        print("I could not guess the book based on your answers.")