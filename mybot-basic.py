import csv
import aiml
import wikipedia
import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from nltk.inference import ResolutionProver
import time
time.clock = time.perf_counter

from similarity_function import findSimilarity
from functions import getAuthorFromTitle
from guess_book import guessing_game

from nltk.sem import Expression
from nltk.inference import ResolutionProver
import nltk
read_expr = Expression.fromstring

# Load KB 
kb = []
with open("KB.csv", "r") as f:
    lines = f.readlines()[1:]

for line in lines:
    clean_line = line.strip().strip('"')
    if clean_line:
        kb.append(read_expr(clean_line))

# CNN Model
model = keras.models.load_model("book_model.keras")

# Load Q/A pairs
data = {
  "Questions": [],
  "Answers": []
}

fileName = "QA.csv"
QAKB = []
with open(fileName, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for item in csvreader:  # Read rows
        if len(item) < 2:  # skip invalid or empty lines
            continue
        data["Questions"].append(item[0])
        data["Answers"].append(item[1])

# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.verbose(False) # get rid of the warning
kern.bootstrap(learnFiles="mybot-basic.xml")
print(f"Loaded {kern.numCategories()} AIML categories.\n")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!\n")
print("Some things you could say:")
print("1. I know that {Book Name} is {Genre Name}")
print("1.2 I Know that {Author Name} is the Author of {Book Name}")
print("2. Check that {Book Name} is {Genre Name}")
print("2.2 Check that {Author Name} is the Author of {Book Name}")
print("3 Author of {Name of Book}")
print("4. Ask to play the guessing game")
print("5. Is this book cover typography or retro style?")

# Main loop
while True:
    # get user input
    try:
        userInput = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'

    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput.upper())  # returns the <template> 

        if answer and answer.startswith('#'):
            params = answer[1:].split('$')
            cmd = int(params[0])

            if cmd == 0:
                print(params[1]) 
                break

            elif cmd == 1:
                query = params[1]
                try:
                    summary = wikipedia.summary(query, sentences=2)
                    print(summary)
                except:
                    print("Sorry, I couldn't find information on that.")
                continue

            elif cmd == 2:
                print("Here are some things you can try:")
                print("1. I know that Harry Potter is fantasy")
                print("2. Check that Harry Potter is fantasy")
                print("3. Author of The Hobbit")
                print("4. Play the guessing game")
                continue

        else:
            print(answer)

    # FOL 
    match = re.match(r"I know that (.+?) is (.+)", userInput, re.IGNORECASE)
    if match:
        subject = match.group(1).replace(" ", "")
        raw_category = match.group(2).lower().strip()

        author_match = re.match(r"(?:the\s+)?author of (.+)", raw_category)

        if author_match:
            book = author_match.group(1).replace(" ", "")
            new_expr = read_expr(f"Wrote({subject},{book})")
            neg_expr = read_expr(f"-Wrote({subject},{book})")

            if ResolutionProver().prove(neg_expr, kb):
                print("That contradicts what I already know.")
            else:
                kb.append(new_expr)
                print(f"OK, I will remember that {match.group(1)} wrote {author_match.group(1)}.")
        else:
            category = raw_category.replace("-", "").replace(" ", "").capitalize()
            new_expr = read_expr(f"{category}({subject})")
            neg_expr = read_expr(f"-{category}({subject})")

            if ResolutionProver().prove(neg_expr, kb):
                print("That contradicts what I already know.")
            else:
                kb.append(new_expr)
                print(f"OK, I will remember that {match.group(1)} is {raw_category}.")
        continue

    match2 = re.match(r"Check that (.+?) is (.+)", userInput, re.IGNORECASE)
    if match2:
        obj = match2.group(1).replace(" ", "")
        raw_category = match2.group(2).lower().strip()

        # Handle author-of-book separately
        author_match = re.match(r"(?:the\s+)?author of (.+)", raw_category)
        if author_match:
            book = author_match.group(1).replace(" ", "")
            subject = match2.group(1).replace(" ", "")

            goal = read_expr(f"Wrote({subject},{book})")
            neg_goal = read_expr(f"-Wrote({subject},{book})")

            display = f"{match2.group(1)} wrote {author_match.group(1)}" # Combine author and book
        else:
            # Original handling for other categories
            category = raw_category.replace("-", "").replace(" ", "").capitalize()
            display_category = raw_category.capitalize()
            display_obj = match2.group(1)

            goal = read_expr(f"{category}({obj})")
            neg_goal = read_expr(f"-{category}({obj})")

        if ResolutionProver().prove(goal, kb):
            print(f"Yes, it is correct that {display_obj} is {display_category}.")
        elif ResolutionProver().prove(neg_goal, kb):
            print(f"No, that is incorrect.")
        else:
            print("I don't know.")
        continue

    # books api
    if "author of" in userInput.lower():
        title = userInput.lower().replace("author of", "").strip()
        result = getAuthorFromTitle(title)
        print(result)
        continue

    if "guessing game" in userInput.lower():
        print("entering guessing game")
        guessing_game(kb)
        continue
 
    # Image classification
    if "typography" in userInput.lower() and "retro" in userInput.lower():
        root = Tk()
        root.withdraw()
        # Open file dialog
        file_path = askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        print("Selected image:", file_path)
                         
        img = tf.keras.utils.load_img(
            file_path,
            target_size=(224,224)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0   # same normalization as training
        img_array = np.expand_dims(img_array, axis=0)  
        prediction = model.predict(img_array)

        if prediction > 0.7:
            print("Typography-Focused")
        elif prediction < 0.3:
            print("Retro")
        else:
            print("Unknown / Neither")
        continue

    # If all else fails, use similarity from CSV
    if not answer or answer.strip() == "": # AIML didn't match
        sim_result = findSimilarity(data, userInput)
        if sim_result:
            score = sim_result[0]
            response = sim_result[1]
            print('similarity score: ',score)
            if score > 0.3:   # threshold 
                answer = response
            else:
                answer = "I’m not sure about that. Can you rephrase?"
        else:
            answer = "I did not get that, please try again."
        print(answer)
   
    