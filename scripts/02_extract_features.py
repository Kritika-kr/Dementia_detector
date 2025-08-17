import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim.downloader as api
import spacy
import re
import joblib


df = pd.read_csv('data/preprocessed_data.csv')
texts = df['Processed_Text'].astype(str).tolist()


os.makedirs("features", exist_ok=True)
os.makedirs("outputs/vectorizers", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=300)
tfidf_features = tfidf.fit_transform(texts).toarray()
np.save("features/features_tfidf.npy", tfidf_features)
joblib.dump(tfidf, "outputs/vectorizers/tfidf_vectorizer.pkl")


print("Extracting CountVectorizer features...")
count_vec = CountVectorizer(max_features=300)
count_features = count_vec.fit_transform(texts).toarray()
np.save("features/features_count.npy", count_features)
joblib.dump(count_vec, "outputs/vectorizers/count_vectorizer.pkl")


print("Loading Word2Vec model...")
word2vec_model = api.load("word2vec-google-news-300")
word2vec_model.save("outputs/models/word2vec_model.bin")

def get_avg_embedding(text, model, dim=300):
    words = text.split()
    valid = [w for w in words if w in model]
    return np.mean([model[w] for w in valid], axis=0) if valid else np.zeros(dim)

print("Extracting Word2Vec embeddings...")
word2vec_features = np.array([get_avg_embedding(text, word2vec_model) for text in tqdm(texts)])
np.save("features/features_word2vec.npy", word2vec_features)
np.save("features/features_embed_word2vec.npy", word2vec_features)


print("Loading GloVe model...")
glove_model = api.load("glove-wiki-gigaword-300")


print("Extracting GloVe embeddings...")
glove_features = np.array([get_avg_embedding(text, glove_model) for text in tqdm(texts)])
np.save("features/features_embed_glove.npy", glove_features)


print("Extracting sentence embeddings...")
nlp = spacy.load("en_core_web_md")
sentence_embed = np.array([nlp(text).vector for text in tqdm(texts)])
np.save("features/features_sentence_embed.npy", sentence_embed)


print("Extracting word frequency features...")
word_freq = CountVectorizer()
freq_features = word_freq.fit_transform(texts).toarray()
np.save("features/features_frequency.npy", freq_features)
joblib.dump(word_freq, "outputs/vectorizers/frequency_vectorizer.pkl") 


hesitation_words = {'uh', 'um', 'erm', 'ah', 'eh', 'hmm'}

def count_hesitations(text):
    return sum(1 for w in text.lower().split() if w in hesitation_words)

print("Extracting hesitation features...")
hesitation_features = np.array([[count_hesitations(text)] for text in texts])
np.save("features/features_hesitation.npy", hesitation_features)

def extract_linguistic_features(text):
    doc = nlp(text)
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    avg_token_length = np.mean([len(token.text) for token in doc]) if num_tokens > 0 else 0
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    return [num_tokens, num_sentences, avg_token_length, noun_count, verb_count]

print("Extracting linguistic features...")
linguistic_features = np.array([extract_linguistic_features(text) for text in tqdm(texts)])
np.save("features/features_linguistic.npy", linguistic_features)

print("\nAll features extracted and saved in 'features/' folder.")
print("Vectorizers and models saved in 'outputs/vectorizers/' and 'outputs/models/'")
