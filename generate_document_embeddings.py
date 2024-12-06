import os
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

model = Doc2Vec.load('doc2vec_model.d2v')
vector = model.infer_vector(["system", "response"])
print(vector)