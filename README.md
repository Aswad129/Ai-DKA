# Sistem Klasifikasi Sentimen Komentar MBG (TF-IDF + SVM)

## Instalasi
    pip install -r requirements.txt

## Alur (jalankan berurutan)
1. **Ambil data**
       set YOUTUBE_API_KEY=KUNCI_API_KAMU        (Linux/Mac: export ...)
       python ambil_komentar.py VIDEO_ID1 VIDEO_ID2
   -> `komentar_mentah.csv`. API key: Google Cloud Console > aktifkan "YouTube Data API v3".
2. **EDA + praproses + label lexicon**
       python siapkan_data.py
   -> `komentar_berlabel.csv`, `sampel_validasi.csv`, folder `grafik/` (untuk bab EDA).
3. **Validasi manual (penting):** buka `sampel_validasi.csv`, isi kolom `label_manual`
   dengan positif / netral / negatif (minimal 2 anggota, bagi dua lalu silang cek).
4. **Latih & evaluasi**
       python latih.py
   -> metrik per kelas, `hasil_perbandingan.csv`, `grafik/confusion_matrix.png`, `model_sentimen_mbg.joblib`.
5. **Prototipe**
       streamlit run app.py

## Catatan
- Lexicon di `praproses.py` hanya STARTER (kecil). Ganti dengan lexicon InSet agar label lebih akurat.
- Tambah kata tidak baku di `NORMALISASI` sesuai komentar yang kalian temukan.
- Kode sudah diuji dengan data sintetis; angka hasil sungguhan baru ada setelah memakai data YouTube asli.
