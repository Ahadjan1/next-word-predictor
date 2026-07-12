import streamlit as st
import pickle
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_assets():
    model = load_model('lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

def predict_top_k(text, k=3):
    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(padded, verbose=0)[0]
    top_idx = pred.argsort()[-k:][::-1]
    return [(index_to_word.get(i, ""), float(pred[i])) for i in top_idx if index_to_word.get(i, "")]

st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #f5f3ff 0%, #eef2f7 100%);
    }

    .hero {
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 52px;
        font-weight: 700;
        background: linear-gradient(90deg, #7F5AF0, #2CB67D, #7F5AF0);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
        margin-bottom: 0;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .hero-subtitle {
        color: #6b6b80;
        font-size: 16px;
        font-weight: 400;
        margin-top: 4px;
    }

    .badge-row { text-align: center; margin: 18px 0 30px 0; }
    .badge {
        display: inline-block;
        background: rgba(127, 90, 240, 0.08);
        border: 1px solid rgba(127, 90, 240, 0.25);
        color: #6246ea;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 0 5px;
        letter-spacing: 0.3px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(127, 90, 240, 0.12);
        border-radius: 20px;
        padding: 28px 30px;
        box-shadow: 0 4px 24px rgba(127, 90, 240, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(div.stButton) {
        border-color: rgba(127, 90, 240, 0.2) !important;
        border-radius: 18px !important;
    }

    .result-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 600;
        color: #1a1a2e;
        line-height: 1.7;
        text-align: center;
        padding: 10px 0;
    }
    .seed-part { color: #6b6b80; }
    .generated-part { color: #2CB67D; }

    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1px solid rgba(127, 90, 240, 0.2) !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    .stTextInput input:focus {
        border: 1px solid #7F5AF0 !important;
        box-shadow: 0 0 0 1px #7F5AF0 !important;
    }

    .stSlider label { color: #3a3a4a !important; font-weight: 500; }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3.2em;
        font-weight: 700;
        font-size: 15px;
        background: linear-gradient(90deg, #7F5AF0, #6246ea);
        color: white;
        border: none;
        transition: all 0.25s ease;
        letter-spacing: 0.3px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(127, 90, 240, 0.4);
    }

    .prob-bar-label {
        color: #3a3a4a;
        font-size: 13.5px;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }

    .footer-note {
        text-align: center;
        color: #8a8a9a;
        font-size: 12.5px;
        margin-top: 40px;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <p class="hero-title">🧠 Next Word Predictor</p>
        <p class="hero-subtitle">An LSTM language model that continues your sentence — one word at a time</p>
    </div>
    <div class="badge-row">
        <span class="badge">🧬 LSTM Network</span>
        <span class="badge">☁️ Trained on Google Colab</span>
        <span class="badge">📚 10,000-word Vocabulary</span>
    </div>
""", unsafe_allow_html=True)

if "seed_input" not in st.session_state:
    st.session_state.seed_input = ""

left, right = st.columns([1.3, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("##### ✏️ Enter a starting phrase")

        seed_text = st.text_input(
            "seed", value=st.session_state.seed_input,
            placeholder="e.g. i am", label_visibility="collapsed"
        )

        examples = ["i am", "the world is", "love is", "i think that", "life is a"]
        ex_cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.seed_input = ex
                st.rerun()

        n_words = st.slider("Words to generate", min_value=1, max_value=30, value=10)
        generate_clicked = st.button("✨ Generate Text", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown("##### 🔮 Top predictions (next word)")
        preview_box = st.empty()
        if seed_text.strip():
            preds = predict_top_k(seed_text.lower(), k=5)
            with preview_box.container():
                for word, prob in preds:
                    st.markdown(f"""
                        <div class="prob-bar-label"><span>{word}</span><span>{prob*100:.1f}%</span></div>
                    """, unsafe_allow_html=True)
                    st.progress(min(float(prob), 1.0))
        else:
            preview_box.markdown(
                "<p style='color:#9a9aab; font-size:14px;'>Start typing to see live next-word predictions...</p>",
                unsafe_allow_html=True
            )

if generate_clicked:
    if seed_text.strip() == "":
        st.warning("⚠️ Please enter a starting phrase first.")
    else:
        placeholder = st.empty()
        generated_words = []

        with st.spinner("Generating..."):
            text_so_far = seed_text.lower()
            for _ in range(n_words):
                top = predict_top_k(text_so_far, k=1)
                if not top:
                    break
                next_word = top[0][0]
                text_so_far += " " + next_word
                generated_words.append(next_word)
                placeholder.markdown(
                    "<div class=\"glass-card\"><div class=\"result-text\">"
                    f"<span class=\"seed-part\">{seed_text}</span>"
                    f"<span class=\"generated-part\"> {' '.join(generated_words)}</span>"
                    "</div></div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.12)

        with st.expander("ℹ️ About this model"):
            st.write("""
            This model was trained using an **LSTM (Long Short-Term Memory)** network 
            on **Google Colab** with GPU acceleration. It predicts the next word in a 
            sequence based on patterns learned from a quotes dataset. Since this is a 
            large multi-class problem (~10,000 possible next words), generated text may 
            occasionally be grammatically imperfect — this is expected for a language 
            model trained on a limited dataset and number of epochs.
            """)

st.markdown('<p class="footer-note">Built with Streamlit · LSTM Text Generation · Trained on Google Colab</p>', unsafe_allow_html=True)