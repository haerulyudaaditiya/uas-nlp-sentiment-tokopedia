import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import hstack
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Caching Resources untuk Kecepatan Loading
@st.cache_resource
def load_nlp_resources():
    # Inisialisasi Stopwords Sastrawi (Negasi & Kontras Dipertahankan)
    stopword_factory = StopWordRemoverFactory()
    base_stopwords = set(stopword_factory.get_stop_words())

    negation_words = {
        "tidak", "tak", "tiada", "bukan", "belum", "tanpa", "kurang",
        "jangan", "tidaklah", "bukannya", "tapi", "tetapi", "namun",
        "melainkan", "sebaliknya"
    }

    stopwords_list = {word for word in base_stopwords if word not in negation_words}

    # Stopwords kustom yang netral untuk e-commerce
    custom_stopwords = {
        "barang", "produk", "beli", "pesan", "order", "seller", "toko",
        "admin", "respon", "kirim", "kurir", "pengiriman", "paket", "packing",
        "nya"
    }
    stopwords_list.update(custom_stopwords)

    # Kamus Slang Bahasa Indonesia untuk Normalisasi
    slang_dict = {
        "yg": "yang", "dg": "dengan", "dgn": "dengan", "kalo": "kalau",
        "jgn": "jangan", "sdh": "sudah", "udah": "sudah", "bgt": "banget",
        "tdk": "tidak", "ga": "tidak", "gak": "tidak", "tp": "tapi",
        "kmrn": "kemarin", "dpt": "dapat", "brg": "barang", "skrg": "sekarang",
        "krn": "karena", "blm": "belum", "bbrp": "beberapa", "utk": "untuk",
        "aja": "saja", "lah": "saja", "sya": "saya"
    }

    return stopwords_list, slang_dict

@st.cache_resource
def load_ml_models():
    # Memuat TF-IDF Vectorizer, Rating Encoder, dan Model Klasifikasi Sentimen
    models_dir = "models"
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    rating_encoder = joblib.load(os.path.join(models_dir, "rating_encoder.pkl"))
    sentiment_model = joblib.load(os.path.join(models_dir, "best_sentiment_model.pkl"))
    return vectorizer, rating_encoder, sentiment_model

# Memuat resources
stopwords_list, slang_dict = load_nlp_resources()
vectorizer, rating_encoder, sentiment_model = load_ml_models()

# Deteksi tipe model secara dinamis untuk laporan performa
if type(sentiment_model).__name__ == "MultinomialNB":
    model_friendly_name = "Multinomial Naive Bayes"
    model_f1_score = "98.43%"
elif type(sentiment_model).__name__ == "LinearSVC":
    model_friendly_name = "Linear SVC"
    model_f1_score = "98.33%"
elif type(sentiment_model).__name__ == "LogisticRegression":
    model_friendly_name = "Logistic Regression"
    model_f1_score = "98.33%"
else:
    model_friendly_name = type(sentiment_model).__name__
    model_f1_score = "98.33%"

# 3. Fungsi Preprocessing Teks (Sesuai Pipeline Training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_and_remove_stopwords(text, slang_dict, stopwords_list):
    tokens = text.split()
    normalized_tokens = [slang_dict.get(word, word) for word in tokens]
    filtered_tokens = [word for word in normalized_tokens if word not in stopwords_list and len(word) > 2]
    return filtered_tokens

def preprocess_pipeline(raw_text):
    step1_clean = clean_text(raw_text)
    step2_tokens = normalize_and_remove_stopwords(step1_clean, slang_dict, stopwords_list)
    processed_text = " ".join(step2_tokens)
    return {
        "raw": raw_text,
        "clean": step1_clean,
        "tokens": step2_tokens,
        "final": processed_text
    }

# 4. Custom CSS untuk Tampilan Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .card {
        background-color: var(--secondary-background-color);
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.15);
        padding-bottom: 8px;
    }

    .result-badge {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        display: inline-block;
    }

    .badge-positive {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }

    .badge-negative {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-color);
    }

    .metric-value {
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.8;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# 5. Sidebar - Informasi Proyek & Model
st.sidebar.markdown("<h3 style='color:var(--text-color);'>UAS Pemrosesan Bahasa Alami</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:var(--text-color); opacity:0.7; font-size:0.9rem;'>Klasifikasi Sentimen Ulasan Tokopedia</p>", unsafe_allow_html=True)

# 6. Main Panel
st.markdown("<h1 class='main-title'>Analisis Sentimen Ulasan Tokopedia</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-title'>Klasifikasi sentimen positif atau negatif dari ulasan pelanggan menggunakan representasi fitur gabungan TF-IDF teks ulasan dan Customer Rating dengan model klasifikasi terbaik ({model_friendly_name}).</p>", unsafe_allow_html=True)

# Contoh ulasan untuk memudahkan pengujian
st.markdown("<h4 style='color:var(--text-color);'>Pilih Contoh Ulasan atau Tulis Sendiri</h4>", unsafe_allow_html=True)

samples = {
    "Pilih ulasan contoh...": {
        "text": "",
        "rating": 5
    },
    "Contoh 1: Barang bagus banget, pengemasan sangat rapi dan pengiriman super cepat! Sangat puas belanja di sini. (Rating 5)": {
        "text": "Barang bagus banget, pengemasan sangat rapi dan pengiriman super cepat! Sangat puas belanja di sini.",
        "rating": 5
    },
    "Contoh 2: Sangat kecewa! Barang yang datang pecah dan tidak bisa dipakai. Kapok belanja di toko ini. (Rating 1)": {
        "text": "Sangat kecewa! Barang yang datang pecah dan tidak bisa dipakai. Kapok belanja di toko ini.",
        "rating": 1
    },
    "Contoh 3: Takut sekali pas kurirnya bawa paketnya basah kuyup karena hujan lebat, untung di dalam dilapisi plastik tebal jadi barang aman. (Rating 4)": {
        "text": "Takut sekali pas kurirnya bawa paketnya basah kuyup karena hujan lebat, untung di dalam dilapisi plastik tebal jadi barang aman.",
        "rating": 4
    },
    "Contoh 4: Sudah bayar mahal tapi respon admin lambat sekali dan kiriman belum sampai juga padahal butuh cepat. (Rating 2)": {
        "text": "Sudah bayar mahal tapi respon admin lambat sekali dan kiriman belum sampai juga padahal butuh cepat.",
        "rating": 2
    },
    "Contoh 5: Suka sekali dengan bajunya, bahannya sangat lembut dan pas di badan. Terima kasih seller. (Rating 5)": {
        "text": "Suka sekali dengan bajunya, bahannya sangat lembut dan pas di badan. Terima kasih seller.",
        "rating": 5
    }
}

selected_sample = st.selectbox("Gunakan ulasan contoh berikut untuk pengujian cepat:", list(samples.keys()))
input_text_placeholder = samples[selected_sample]["text"]
input_rating_placeholder = samples[selected_sample]["rating"]

# Grid layout untuk input
col_input, col_info = st.columns([2, 1])

with col_input:
    # Form input ulasan
    user_review = st.text_area(
        "Masukkan ulasan pelanggan di bawah ini:",
        value=input_text_placeholder,
        height=120,
        placeholder="Tulis ulasan produk di sini..."
    )

with col_info:
    # Slider rating pelanggan
    user_rating = st.slider(
        "Pilih Rating Pelanggan (Skala 1 - 5):",
        min_value=1,
        max_value=5,
        value=int(input_rating_placeholder),
        step=1,
        help="Rating yang diberikan pelanggan membantu model memprediksi sentimen secara lebih akurat."
    )

# Tombol analisa
analyze_btn = st.button("Analisis Ulasan", type="primary")

if analyze_btn or user_review.strip():
    if not user_review.strip():
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses teks dan melakukan klasifikasi..."):
            # A. Jalankan Preprocessing
            prep_res = preprocess_pipeline(user_review)

            # Jika teks kosong setelah diproses
            if not prep_res["final"].strip():
                st.error("Teks input tidak mengandung kata yang dapat diproses setelah penyaringan stopword. Silakan masukkan ulasan yang lebih panjang.")
            else:
                # B. Ekstraksi Fitur TF-IDF
                X_text_vec = vectorizer.transform([prep_res["final"]])

                # C. Encoding Customer Rating
                X_rating_vec = rating_encoder.transform(pd.DataFrame([[user_rating]], columns=["Customer Rating"]))

                # D. Menggabungkan Fitur
                X_combined = hstack([X_text_vec, X_rating_vec])

                # E. Prediksi Sentimen dengan fallback probabilitas robust
                sent_classes = list(sentiment_model.classes_)
                pos_idx = sent_classes.index("Positive")
                neg_idx = sent_classes.index("Negative")

                if hasattr(sentiment_model, "predict_proba"):
                    probs = sentiment_model.predict_proba(X_combined)[0]
                    prob_positive = probs[pos_idx]
                    prob_negative = probs[neg_idx]
                else:
                    sent_decision = sentiment_model.decision_function(X_combined)[0]
                    # Formula sigmoid untuk mapping (-inf, inf) -> (0, 1)
                    prob_positive = 1.0 / (1.0 + np.exp(-sent_decision))
                    prob_negative = 1.0 - prob_positive

                pred_sentiment = "Positive" if prob_positive >= 0.5 else "Negative"
                sent_confidence = prob_positive if pred_sentiment == "Positive" else prob_negative
                
                # Menyiapkan list probabilitas sesuai urutan kelas asli untuk chart
                sent_probs = [0.0, 0.0]
                sent_probs[pos_idx] = prob_positive
                sent_probs[neg_idx] = prob_negative

                # Tampilkan Hasil
                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    st.markdown("""
                    <div class='card'>
                        <div class='card-title'>Hasil Prediksi Sentimen</div>
                    """, unsafe_allow_html=True)

                    # Badge Kelas Sentimen
                    badge_class = "badge-positive" if pred_sentiment == "Positive" else "badge-negative"
                    st.markdown(f"<div class='result-badge {badge_class}'>{pred_sentiment}</div>", unsafe_allow_html=True)

                    # Persentase Keyakinan
                    st.markdown(f"<p style='font-size:1.1rem; color:var(--text-color); opacity:0.8;'>Tingkat Keyakinan: <b>{sent_confidence*100:.2f}%</b></p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_res2:
                    st.markdown("""
                    <div class='card'>
                        <div class='card-title'>Probabilitas Kelas Sentimen</div>
                    """, unsafe_allow_html=True)

                    # Progress Bars untuk semua kelas sentimen
                    for label, prob in zip(sent_classes, sent_probs):
                        st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:2px;'><span class='metric-label'>{label}</span><span class='metric-value'>{prob*100:.2f}%</span></div>", unsafe_allow_html=True)
                        st.progress(float(prob))

                    st.markdown("</div>", unsafe_allow_html=True)


