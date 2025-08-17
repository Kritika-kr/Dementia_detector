import streamlit as st
import joblib
import numpy as np
from extract_combined_features_single import extract_features

# Load the trained Voting model
model_path = "outputs/models/ensemble_lasso_rf_svm.pkl"
model = joblib.load(model_path)

# Streamlit UI
st.set_page_config(page_title="Dementia Detection (Voting Model)", layout="centered")
st.title("Dementia Detection App")
st.subheader("Using Voting Ensemble on Combined Features")

st.markdown("""
This app detects dementia from your input speech transcript using a Voting Classifier trained on combined linguistic and embedding-based features.
""")

# Text Input
user_input = st.text_area("Enter speech transcript (e.g., patient's response to a task):", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text input.")
    else:
        with st.spinner("Extracting features and predicting..."):
            # Extract features
            features = extract_features(user_input)

            # Predict
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            # Display result
            label_map = {0: "Control", 1: "Dementia"}
            st.success(f"**Prediction:** {label_map[prediction]}")
            st.info(f"**Confidence:** Control: {proba[0]:.2f}, Dementia: {proba[1]:.2f}")
