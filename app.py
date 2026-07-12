import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model, tokenizer, and max_len
model = load_model('lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)

# Reverse mapping (index -> word)
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# Predict next word
def predictor(model, tokenizer, text, max_len):
    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(padded, verbose=0)
    predicted_index = np.argmax(pred)
    return index_to_word.get(predicted_index, "")

# Generate multiple words
def generate_text(seed_text, n_words):
    text = seed_text
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, text, max_len)
        text += " " + next_word
    return text

# ---------- Page Config ----------
st.set_page_config(page_title="AI Quote Generator", page_icon="✍️", layout="centered")

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #4D96FF22;
        border: 2px solid #4D96FF;
        color: #4D96FF;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        background-color: #4D96FF;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">✍️ AI Quote Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a starting phrase and let the LSTM model continue it.</p>', unsafe_allow_html=True)

seed_text = st.text_input("Starting phrase (seed text)", placeholder="e.g. i am")
n_words = st.slider("Number of words to generate", min_value=1, max_value=30, value=10)

if st.button("✨ Generate Quote"):
    if seed_text.strip() == "":
        st.warning("⚠️ Please enter a starting phrase.")
    else:
        with st.spinner("Generating..."):
            generated = generate_text(seed_text.lower(), n_words)
        st.markdown(f'<div class="result-box">"{generated}"</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit | LSTM-based Text Generation")