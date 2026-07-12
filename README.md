# ✍️ Next Word Predictor (LSTM-based Text Generation)

An LSTM-based language model that predicts and generates the next word(s) given a starting phrase — trained on a quotes dataset and deployed as an interactive Streamlit app.

**🔗 Live Demo:** [next-word-predictor-l.streamlit.app](https://next-word-predictor-l.streamlit.app/)

---

## 📌 Overview

This project builds a word-level language model using an **LSTM (Long Short-Term Memory)** network. Given a seed phrase (e.g. *"i am"*), the model predicts the most likely next word and can generate a full sequence of words to continue the phrase — one word at a time, feeding each prediction back into itself.

---

## 🚀 Features

- Real-time text generation from any starting phrase
- Live "next word" predictions with probability scores as you type
- Quick-start example phrases
- Adjustable number of words to generate
- Clean, interactive Streamlit UI

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Model Training | Google Colab (GPU) |
| Deep Learning Framework | TensorFlow / Keras |
| Architecture | Embedding → LSTM → Dense (Softmax) |
| Deployment | Streamlit Community Cloud |

---

## 🧠 Model Pipeline

1. **Data Loading** — Quotes dataset (`quote_dataset.csv`) loaded into pandas
2. **Text Cleaning** — Lowercasing, punctuation removal
3. **Tokenization** — Keras `Tokenizer` (vocab_size = 10,000) to map words to indices
4. **Sequence Generation** — Created (prefix → next word) training pairs from each quote
5. **Padding** — Pre-padded sequences to a fixed `max_len`
6. **One-Hot Encoding** — Target words one-hot encoded for multi-class classification (10,000 classes)
7. **Model Architecture:**
   - `Embedding` layer (embedding_dim = 50)
   - `LSTM` layer (units = 128)
   - `Dense` output layer with `softmax` activation
8. **Training** — Compiled with Adam optimizer and categorical crossentropy loss, trained for up to 100 epochs (batch_size = 128) on Google Colab with GPU acceleration
9. **Text Generation** — A custom `predictor()` function predicts the next word, and `generate_text()` repeatedly calls it to generate a sequence of words from a seed phrase

---

## ⚠️ Notes on Training

- Model trained on **Google Colab** using a **T4 GPU**
- A **SimpleRNN** baseline model was also built for comparison — LSTM was chosen for deployment due to its ability to better retain long-term context in sequences
- Since this is a **multi-class classification problem with ~10,000 possible next-word classes**, accuracy in the early epochs is expected to be low — much higher than random chance, but far from the near-100% accuracy seen in simpler binary classification tasks
- Generated text may occasionally be grammatically imperfect, which is expected behavior for a word-level language model trained on a limited dataset and number of epochs

---

## 📂 Project Structure

```
next-word-predictor/
│
├── app.py               # Streamlit application
├── lstm_model.h5         # Trained LSTM model
├── tokenizer.pkl          # Fitted Keras Tokenizer
├── max_len.pkl            # Max sequence length used during training
├── requirements.txt
├── runtime.txt            # Pins Python version (3.11) for TensorFlow compatibility
└── README.md
```

---

## 💻 Run Locally

```bash
git clone https://github.com/Ahadjan1/next-word-predictor.git
cd next-word-predictor
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** This project requires **Python 3.9–3.12** for TensorFlow compatibility. If your default Python version is newer (e.g. 3.13+), create a virtual environment with a supported version:
> ```bash
> py -3.11 -m venv venv
> venv\Scripts\activate      # Windows
> pip install -r requirements.txt
> ```

---

## 👤 Author

**Ahad Jan**
- GitHub: [@Ahadjan1](https://github.com/Ahadjan1)
- LinkedIn: [ahad-jan](https://www.linkedin.com/in/ahad-jan-b8ba872b4/?skipRedirect=true)
