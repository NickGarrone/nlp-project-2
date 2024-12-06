import os
import uuid
import os
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json


def extract_pdfs(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def extract_wikipedia_texts_from_file(file_path):
    data = None
    with open(f"training_data_raw/{file_path}", "r", encoding="utf-8") as f:
        data = json.load(f)
    for article in data:
        yield article['text']

#Strip all newlines and headers
def preprocess_text(text: string):
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text[1000:]
    return text

def write_training_data(text):
    filename = str(uuid.uuid1()) + ".txt"
    with open(f"training_data/{filename}", "a", encoding="utf-8") as f:
        f.write(text)
        f.close()

def main(directory_path, limit=100):
    processed = 0
    for filename in os.listdir(directory_path):
        print(filename)
        print(processed)
        if filename.endswith('.json'):
            for article in extract_wikipedia_texts_from_file(filename):
                text = preprocess_text(article)
                write_training_data(text)
                processed += 1
        elif filename.endswith('.pdf'):
            article = extract_pdfs(filename)
            text = preprocess_text(article)
            write_training_data(text)
            processed += 1
        if processed >= limit:
            break

main("training_data_raw", 1000000)
