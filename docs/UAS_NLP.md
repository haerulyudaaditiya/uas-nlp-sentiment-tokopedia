UJIAN AKHIR SEMESTER (UAS)
Mata Kuliah: Pemrosesan Bahasa Alami (Natural Language Processing)

Deskripsi Tugas
Mahasiswa diminta menyelesaikan sebuah proyek NLP secara individu dengan memilih dataset
teks yang relevan dan membangun solusi berbasis Natural Language Processing menggunakan
Python. Seluruh proses harus didokumentasikan dalam laporan dan diimplementasikan dalam
sebuah aplikasi sederhana.

KETENTUAN UMUM

1.  Dataset harus diperoleh dari sumber publik seperti:

o  Kaggle
o  UCI Machine Learning Repository
o  Hugging Face Dataset
o  Data.gov
o  Portal Open Data lainnya

2.  Dataset minimal memiliki:

o  500 dokumen/kalimat/data teks.
o  Memiliki label/target untuk klasifikasi atau tujuan NLP lainnya.
o  Data dapat berupa ulasan (review), berita, komentar media sosial, tweet, artikel,

atau dokumen teks lainnya.

3.  Bahasa pemrograman yang digunakan:

o  Python

4.  Topik proyek NLP dapat berupa:

o  Sentiment Analysis
o  Text Classification
o  Spam Detection
o  Hate Speech Detection
o  Fake News Detection
o  Topic Classification
Intent Classification
o
o  Named Entity Recognition (NER)
o  Text Summarization
o  atau topik NLP lainnya.

===================================================
1. PEMILIHAN DAN PEMAHAMAN DATASET
a. Cari dan pilih sebuah dataset NLP yang sesuai.
b. Jelaskan:
Judul dataset
•
Sumber dataset
•
•  Tujuan analisis
Jumlah data
•
Jumlah kelas/label
•
•  Deskripsi setiap atribut

===================================================
2. EXPLORATORY DATA ANALYSIS (EDA) TEKS (20%)
Lakukan eksplorasi data teks menggunakan statistik deskriptif dan visualisasi.
Minimal mencakup:

1.  Analisis Struktur Data

Jumlah dokumen
Jumlah atribut

o
o
o  Tipe data
o  Missing value
o  Distribusi kelas

2.  Visualisasi

o  Distribusi kelas (bar chart)
o  Histogram panjang dokumen
o  Word Cloud
o  Top 20 kata paling sering muncul
o  Visualisasi lain yang relevan

3.  Temuan Penting

o  Karakteristik dataset
o  Ketidakseimbangan kelas
o  Kata-kata dominan
o

Insight awal dari data

Output:

•  Grafik
•

Interpretasi setiap grafik

===================================================
3. PRE-PROCESSING DATA TEKS (15%)
Lakukan tahapan pembersihan dan persiapan data teks.
Minimal meliputi:

1.  Case Folding
2.  Cleaning Text

o  Menghapus URL
o  Menghapus tanda baca
o  Menghapus angka
o  Menghapus karakter khusus

3.  Tokenization
4.  Stopword Removal
5.  Stemming atau Lemmatization
6.  Text Representation
o  Bag of Words
o  TF-IDF
o  Word Embedding (opsional)

7.  Pembagian Data

o  Training Set
o  Testing Set

Jelaskan alasan penggunaan setiap metode.
Output:

•  Hasil preprocessing
•  Contoh data sebelum dan sesudah preprocessing

===================================================
4. PEMBANGUNAN MODEL NLP (20%)
Bangun minimal 2 model machine learning untuk menyelesaikan permasalahan NLP yang dipilih.
Contoh model:
Klasifikasi Teks

Logistic Regression

•  Naive Bayes
•
•  Decision Tree
•  Random Forest
•
•  K-Nearest Neighbor (KNN)

Support Vector Machine (SVM)

Deep Learning (Opsional)

LSTM
•
•  GRU
•  CNN untuk Teks
•  BERT / IndoBERT

Tampilkan:

•  Arsitektur/model yang digunakan
•  Parameter yang digunakan
•  Hasil pelatihan

Output:

•  Kode pelatihan model
•  Ringkasan parameter model

===================================================
5. EVALUASI MODEL (15%)
Lakukan evaluasi terhadap seluruh model yang dibangun.
Gunakan:

•  Accuracy
•  Precision
•  Recall
•
•  Confusion Matrix

F1-Score

Analisis:

•  Bandingkan performa seluruh model.
•  Tentukan model terbaik beserta alasannya.
•
Output:

Jelaskan faktor yang memengaruhi hasil model.

•  Tabel perbandingan model
•  Grafik Confusion Matrix
•  Kesimpulan model terbaik

===================================================
6. IMPLEMENTASI MODEL KE APLIKASI (10%)
Implementasikan model terbaik ke dalam aplikasi sederhana.
Pilihan teknologi:
Streamlit
•
Flask
•
•  Django
•  Gradio

Aplikasi minimal memiliki:

Input teks dari pengguna

•
•  Tombol prediksi
•  Hasil prediksi
•  Tingkat kepercayaan (confidence score) jika tersedia

Sertakan:
•
•
•

Screenshot aplikasi
Struktur folder proyek
Source code

===================================================
OUTPUT YANG DIKUMPULKAN

1.  Laporan PDF (minimal 10 halaman)
2.  Source code aplikasi
3.  Dataset

