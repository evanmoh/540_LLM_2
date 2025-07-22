# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Evan Moh - AIPI540 Deep Learning LLM assignment (Duke University)
st.title("NSCLC vs. SCLC Clinical Text Classifier")
st.caption("by Evan Moh — Duke AIPI540 Deep Learning LLM Assignment")

# Load model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('models/nn_model.keras')
    with open('models/nn_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f) 
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Settings 
MAX_LEN = 50

def preprocess(text):
    # Tokenize and pad 
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

def classify_text(text):
    x = preprocess(text)
    prob = model.predict(x)[0][0]
    pred = "NSCLC" if prob < 0.5 else "SCLC"
    confidence = prob if pred == "SCLC" else 1 - prob
    return pred, confidence

# Example clinical texts
examples = {
    "SCLC": [
        "However, some patients with resectable tumors (stage I–IIA) are eligible for curative surgery followed by systemic therapy with or without mediastinal RT.",
        "Ideally, a screening test should detect disease at an early stage when it is still curable.",
        "Frequently, patients present with symptoms of widespread metastatic disease, such as weight loss, debility, bone pain, and neurologic compromise."
    ],
    "NSCLC": [
        "FDG-PET/CT significantly improves targeting accuracy, especially for patients with significant atelectasis and when IV CT contrast is contraindicated.",
        "The presence of a KRAS mutation is prognostic of poor survival when compared to patients with tumors without KRAS mutation.",
        "For patients with an underlying EGFR sensitizing mutation who have been treated with EGFR TKI, minimum appropriate testing includes high-sensitivity evaluation for p.T790M."
    ]
}
# Input
st.subheader("Paste your clinical text below:")

user_text = st.text_area("Enter clinical sentence here:", height=100)

if st.button("Classify Text"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred, confidence = classify_text(user_text)
        st.success(f"Prediction: **{pred}** (Confidence: {confidence:.2f})")

st.divider()
st.subheader("Try with example sentences:")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**SCLC Examples:**")
    for idx, s in enumerate(examples["SCLC"]):
        if st.button(f"Try Example SCLC {idx+1}"):
            pred, confidence = classify_text(s)
            st.info(f"Example {idx+1} classified as: **{pred}** (Confidence: {confidence:.2f})\n\n> {s}")

with col2:
    st.markdown("**NSCLC Examples:**")
    for idx, s in enumerate(examples["NSCLC"]):
        if st.button(f"Try Example NSCLC {idx+1}"):
            pred, confidence = classify_text(s)
            st.info(f"Example {idx+1} classified as: **{pred}** (Confidence: {confidence:.2f})\n\n> {s}")
