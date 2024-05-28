from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
import numpy as np

nlp = spacy.load('en_core_web_md')

# preprocess sentences by tokenizing and removing punctuation
def preprocess(sentence):
    doc = nlp(sentence.lower())
    tokens = [token.text for token in doc if not token.is_punct]
    return tokens

# calculate Jaccard similarity for all pairs of sentences
def calculate_jaccard_similarity(sentence1, sentence2):
    tokens1 = preprocess(sentence1)
    tokens2 = preprocess(sentence2)
    # convert tokens to sets to calculate Jaccard similarity
    set1 = set(tokens1)
    set2 = set(tokens2)
    # compute Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:  # handle division by zero
        return 0.0
    else:
        return intersection / union
