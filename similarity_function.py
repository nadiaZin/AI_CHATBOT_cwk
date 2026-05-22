import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]
    return " ".join(tokens)

def findSimilarity(data, question):
    # Preprocess all questions
    questions = [preprocess(q) for q in data["Questions"]]
    new_question = preprocess(question)

    # Append new question temporarily
    questions.append(new_question)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    
    # Cosine similarity of new question vs existing
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_index = cosine_sim.argmax()
    
    # Return matched Q/A
    score = cosine_sim[0][best_index]
    return [score, data["Answers"][best_index]]