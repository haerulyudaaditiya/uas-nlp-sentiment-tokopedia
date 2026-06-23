# Proyek UAS Pemrosesan Bahasa Alami (NLP)
## Klasifikasi Sentimen Ulasan Produk Tokopedia (PRDECT-ID Dataset)

Proyek ini dibangun untuk memenuhi persyaratan Ujian Akhir Semester (UAS) Pemrosesan Bahasa Alami. Tujuan proyek ini adalah melakukan klasifikasi sentimen (Positif dan Negatif) pada ulasan produk Tokopedia berbahasa Indonesia dengan menggunakan representasi TF-IDF dan model pembelajaran mesin (Machine Learning) yang menggabungkan fitur teks ulasan dan Customer Rating.

---

## 1. Struktur Direktori Proyek

```text
UAS/
├── app.py                      # Aplikasi Web Streamlit (UI Interaktif)
├── dataset/
│   └── PRDECT-ID Dataset.csv   # Dataset ulasan Tokopedia (5.400 ulasan)
├── docs/
│   ├── UAS NLP.pdf             # Soal / petunjuk pengerjaan UAS
│   └── UAS_NLP.md              # Versi markdown dokumen petunjuk UAS
├── models/
│   ├── best_sentiment_model.pkl  # Model Klasifikasi Sentimen terbaik (Multinomial Naive Bayes)
│   ├── rating_encoder.pkl        # Customer Rating One-Hot Encoder
│   └── tfidf_vectorizer.pkl      # TF-IDF Vectorizer
├── reports/
│   └── figures/                  # Folder visualisasi matriks evaluasi & EDA (300 DPI)
│       ├── cm_sentiment.png
│       ├── cm_sentiment_linear_svc.png
│       ├── cm_sentiment_logistic_regression.png
│       ├── cm_sentiment_multinomial_naive_bayes.png
│       ├── rating_vs_sentiment.png
│       ├── sentiment_distribution.png
│       ├── word_length_distribution.png
│       ├── wordcloud_negative.png
│       └── wordcloud_positive.png
├── uas_nlp_notebook.ipynb      # Jupyter Notebook lengkap berisi visualisasi & proses training
└── venv/                       # Python Virtual Environment
```

---

## 2. Instalasi dan Persiapan Dependensi

Pastikan python3 dan virtualenv telah terpasang di sistem. Untuk mempersiapkan lingkungan virtual dan dependensi, jalankan perintah berikut di terminal:

```bash
# Mengaktifkan virtual environment
source venv/bin/activate

# Menginstal dependensi yang diperlukan (jika belum terinstal)
pip install pandas numpy scikit-learn Sastrawi joblib streamlit wordcloud matplotlib seaborn jupyter
```

---

## 3. Cara Menjalankan Jupyter Notebook

File `uas_nlp_notebook.ipynb` berisi alur lengkap dari awal sampai akhir, termasuk visualisasi data, pembersihan teks, normalisasi slang/singkatan, penyaringan stopwords selektif (mempertahankan kata negasi dan kontras), pelatihan model, evaluasi performa, dan penyimpanan model terbaik.

Sebagai keputusan rekayasa senior (*Senior-grade design decision*), proses **stemming ditiadaan** dalam pipeline akhir. Hal ini didasarkan pada temuan bahwa stemming Sastrawi mereduksi imbuhan krusial (seperti *ter-* dalam *terbaik*, atau *mengecewakan*) yang bernilai emosional tinggi, sekaligus memperlambat performa komputasi. Tanpa stemming, proses training dan inferensi berjalan secara instan (< 1 md) dan akurasi model meningkat signifikan.

Untuk membuka dan menjalankannya:
1. Pastikan virtual environment aktif.
2. Jalankan perintah jupyter notebook di root direktori proyek:
   ```bash
   jupyter notebook uas_nlp_notebook.ipynb
   ```
3. Lakukan eksekusi cell secara berurutan (*Run All Cells*).

---

## 4. Cara Menjalankan Aplikasi Web Streamlit

Aplikasi Streamlit (`app.py`) menyediakan antarmuka web interaktif yang ramah pengguna. Aplikasi memuat model klasifikasi terbaik yang telah disimpan untuk melakukan prediksi sentimen.

Untuk menjalankan aplikasi:
1. Pastikan virtual environment aktif.
2. Jalankan perintah berikut di terminal:
   ```bash
   streamlit run app.py
   ```
3. Aplikasi akan terbuka di browser Anda secara otomatis (biasanya di `http://localhost:8501`).
4. Anda dapat menggunakan ulasan sampel yang telah disediakan di menu dropdown atau mengetik ulasan kustom Anda sendiri untuk diuji.

---

## 5. Ringkasan Hasil Evaluasi Model

Model dievaluasi menggunakan pembagian data *Stratified Train-Test Split* (80% Train, 20% Test). Berikut adalah hasil performa model terbaik setelah dioptimalkan (tanpa stemming, memelihara kata negasi, serta menggabungkan TF-IDF teks & Customer Rating):

* **Model Klasifikasi Sentimen Terbaik:** Multinomial Naive Bayes
* **Weighted F1-Score:** **98.43%**

*Catatan: Seluruh plot visualisasi grafik disimpan dalam resolusi tinggi 300 DPI di dalam direktori `reports/figures/` sesuai dengan instruksi pengerjaan UAS.*
