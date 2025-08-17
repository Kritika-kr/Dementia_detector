import streamlit as st
import joblib
import numpy as np
import librosa
import tempfile
import gensim
import spacy
import re

from st_audiorec import st_audiorec  # pip install streamlit-audiorec

# ======================
# Paths
# ======================
AUDIO_MODEL_PATH = "outputs/models/audio_model_boosted.pkl"
TEXT_MODEL_PATH = "outputs/models/ensemble_lasso_rf_svm.pkl"
TFIDF_VECT_PATH = "outputs/vectorizers/tfidf_vectorizer.pkl"
COUNT_VECT_PATH = "outputs/vectorizers/count_vectorizer.pkl"
FREQ_VECT_PATH = "outputs/vectorizers/frequency_vectorizer.pkl"
WORD2VEC_PATH = "outputs/models/word2vec_model.bin"

# ======================
# Load Models
# ======================
audio_model = joblib.load(AUDIO_MODEL_PATH)
text_model = joblib.load(TEXT_MODEL_PATH)

tfidf_vectorizer = joblib.load(TFIDF_VECT_PATH)
count_vectorizer = joblib.load(COUNT_VECT_PATH)
frequency_vectorizer = joblib.load(FREQ_VECT_PATH)

word2vec_model = gensim.models.KeyedVectors.load(WORD2VEC_PATH)
import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-300")
nlp = spacy.load("en_core_web_md")

hesitation_words = {'uh', 'um', 'erm', 'ah', 'eh', 'hmm'}

# ======================
# Helper Functions
# ======================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

def get_avg_embedding(text, model, dim=300):
    words = text.split()
    if hasattr(model, 'key_to_index'):
        valid = [w for w in words if w in model.key_to_index]
    else:
        valid = [w for w in words if w in model]
    return np.mean([model[w] for w in valid], axis=0) if valid else np.zeros(dim)

def count_hesitations(text):
    words = text.lower().split()
    return sum(1 for w in words if w in hesitation_words) / max(len(words),1)

def extract_linguistic_features(text):
    doc = nlp(text)
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    avg_token_length = np.mean([len(token.text) for token in doc]) if num_tokens>0 else 0
    noun_count = sum(1 for token in doc if token.pos_=="NOUN")
    verb_count = sum(1 for token in doc if token.pos_=="VERB")
    return np.array([num_tokens, num_sentences, avg_token_length, noun_count, verb_count])

def extract_text_features(text):
    text = clean_text(text)

    # Vectorizers
    tfidf_feat = tfidf_vectorizer.transform([text]).toarray()
    count_feat = count_vectorizer.transform([text]).toarray()
    freq_feat = frequency_vectorizer.transform([text]).toarray()

    # Embeddings
    word2vec_feat = get_avg_embedding(text, word2vec_model).reshape(1,-1)
    glove_feat = get_avg_embedding(text, glove_model).reshape(1,-1)
    sentence_embed = nlp(text).vector.reshape(1,-1)

    # Hesitation & linguistic
    hesitation_feat = np.array([[count_hesitations(text)]])
    linguistic_feat = extract_linguistic_features(text).reshape(1,-1)

    # Combine all
    combined = np.concatenate([
        tfidf_feat,
        count_feat,
        freq_feat,
        word2vec_feat,
        glove_feat,
        sentence_embed,
        hesitation_feat,
        linguistic_feat
    ], axis=1)
    return combined

def extract_audio_features(audio_path, n_mfcc=13, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    if len(y) < sr:  # pad short audio
        y = np.pad(y, (0, sr - len(y)), mode='constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1)
    ])
    return features.reshape(1, -1)

# ======================
# Streamlit UI
# ======================
st.title("🧠 Dementia Detection: Audio + Text Ensemble")

tab1, tab2 = st.tabs(["🎙 Audio Input", "📝 Text Input"])

# -----------------
# Audio Tab
# -----------------
with tab1:
    st.write("Record or upload audio (WAV/MP3).")
    wav_audio_data = st_audiorec()
    audio_file = st.file_uploader("Or upload an audio file", type=["wav","mp3"])

    audio_data = None
    if wav_audio_data is not None:
        audio_data = wav_audio_data
    elif audio_file is not None:
        audio_data = audio_file.read()

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        features = extract_audio_features(tmp_path)
        audio_pred = audio_model.predict(features)[0]
        audio_proba = audio_model.predict_proba(features)[0]

        st.subheader("Audio Model Prediction")
        st.write(f"Prediction: **{'Dementia' if audio_pred==1 else 'Control'}**")
        st.write(f"Confidence: Dementia {audio_proba[1]*100:.2f}%, Control {audio_proba[0]*100:.2f}%")
        st.bar_chart({"Control": audio_proba[0], "Dementia": audio_proba[1]})

# -----------------
# Text Tab
# -----------------
with tab2:
    st.write("Enter transcript or text for prediction.")
    text_input = st.text_area("Enter text:")

    if st.button("Predict Text"):
        if text_input.strip():
            features = extract_text_features(text_input)
            text_pred = text_model.predict(features)[0]
            text_proba = text_model.predict_proba(features)[0]

            st.subheader("Text Model Prediction")
            st.write(f"Prediction: **{'Dementia' if text_pred==1 else 'Control'}**")
            st.write(f"Confidence: Dementia {text_proba[1]*100:.2f}%, Control {text_proba[0]*100:.2f}%")
            st.bar_chart({"Control": text_proba[0], "Dementia": text_proba[1]})

# -----------------
# Late Fusion (if both inputs exist)
# -----------------
if st.button("Predict Ensemble (Audio + Text)"):
    if audio_data is None or not text_input.strip():
        st.warning("Both audio and text inputs are required for ensemble prediction.")
    else:
        # Audio features
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        audio_features = extract_audio_features(tmp_path)
        audio_proba = audio_model.predict_proba(audio_features)[0]

        # Text features
        text_features = extract_text_features(text_input)
        text_proba = text_model.predict_proba(text_features)[0]

        # Late fusion: simple average of probabilities
        fused_proba = (audio_proba + text_proba) / 2
        fused_pred = np.argmax(fused_proba)

        st.subheader("Ensemble Prediction (Audio + Text)")
        st.write(f"Prediction: **{'Dementia' if fused_pred==1 else 'Control'}**")
        st.write(f"Confidence: Dementia {fused_proba[1]*100:.2f}%, Control {fused_proba[0]*100:.2f}%")
        st.bar_chart({"Control": fused_proba[0], "Dementia": fused_proba[1]})