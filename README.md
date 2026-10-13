# Sistem Klasifikasi Sentimen Komentar YouTube tentang Program MBG

**TF-IDF + Support Vector Machine (LinearSVC)** | Python | Streamlit | YouTube Data API v3

Proyek Project Based Mid Test (UTS), mata kuliah DKA, Informatika, Universitas Mulawarman.
Kelompok: [nomor] | Anggota: [Nama 1], [Nama 2], [Nama 3], [Nama 4], [Nama 5]

---

## Daftar Isi

1. [Tujuan dan latar belakang](#1-tujuan-dan-latar-belakang)
2. [Gambaran sistem](#2-gambaran-sistem)
3. [Struktur proyek](#3-struktur-proyek)
4. [Instalasi](#4-instalasi)
5. [Cara menjalankan](#5-cara-menjalankan)
6. [Penjelasan tiap tahap: apa, bagaimana hitungannya, dan kenapa](#6-penjelasan-tiap-tahap)
7. [Hasil](#7-hasil)
8. [Kenapa hasilnya begini](#8-kenapa-hasilnya-begini)
9. [Keterbatasan](#9-keterbatasan)
10. [Saran pengembangan](#10-saran-pengembangan)
11. [Deploy](#11-deploy)
12. [Pemecahan masalah](#12-pemecahan-masalah)
13. [Referensi](#13-referensi)

---

## 1. Tujuan dan latar belakang

### Latar belakang

Program Makan Bergizi Gratis (MBG) diluncurkan pada 6 Januari 2025 dengan target sekitar 82,9 juta penerima manfaat dan anggaran awal Rp71 triliun. Program ini menuai beragam tanggapan: ada dukungan, tetapi ada pula kritik terkait pengawasan kualitas makanan, transparansi anggaran, dan dugaan kasus keracunan. Kolom komentar YouTube menjadi tempat masyarakat menyampaikan pendapat itu. Karena jumlah komentar bisa mencapai ribuan, membacanya satu per satu tidak efektif. Analisis sentimen otomatis dapat mengelompokkan komentar menjadi **positif, netral, dan negatif**.

### Rumusan masalah

1. Bagaimana membangun pipeline klasifikasi sentimen komentar YouTube tentang MBG?
2. Seberapa baik SVM mengklasifikasikan sentimen positif, netral, dan negatif, terutama pada kelas minoritas?
3. Bagaimana pengaruh penanganan ketidakseimbangan kelas terhadap performa model?

### Tujuan

- Membangun dan mengevaluasi sistem klasifikasi sentimen komentar MBG berbasis TF-IDF dan SVM.
- Mengukur kinerja **per kelas** (precision, recall, F1), bukan akurasi saja.
- Membuat prototipe aplikasi yang dapat memprediksi sentimen komentar baru.

### Apa yang membedakan dari penelitian acuan

Penelitian acuan (Andianto & Februariyanti, 2026) memakai konfigurasi serupa dan melaporkan akurasi 82,19%, tetapi recall kelas positif hanya sekitar 39% dan netral sekitar 34%, dan label dibuat otomatis dengan kamus tanpa validasi manusia. Proyek ini menambahkan evaluasi per kelas dan macro-F1, validasi label manual pada sampel, perbandingan dengan model lain, dan cross-validation 5-fold.

---

## 2. Gambaran sistem

```
ambil_komentar.py  ->  komentar_mentah.csv
        |
        v
siapkan_data.py    ->  komentar_berlabel.csv + sampel_validasi.csv + grafik/
        |
        v   (isi kolom label_manual di sampel_validasi.csv)
latih.py           ->  model_sentimen_mbg.joblib + hasil_perbandingan.csv + grafik/confusion_matrix.png
        |
        v
app.py (Streamlit) ->  demo: ketik komentar -> prediksi sentimen
```

Alur satu komentar saat prediksi:

```
teks mentah -> praproses -> TF-IDF -> LinearSVC (3 skor) -> skor tertinggi = label
```

---

## 3. Struktur proyek

| File                | Fungsi                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `ambil_komentar.py` | Mengambil komentar video YouTube lewat YouTube Data API v3                               |
| `praproses.py`      | Praproses teks dan pelabelan lexicon (dipakai bersama oleh semua skrip)                  |
| `siapkan_data.py`   | EDA, praproses seluruh komentar, pelabelan lexicon, pembuatan grafik dan sampel validasi |
| `latih.py`          | Melatih dan mengevaluasi model, membandingkan dengan model lain, menyimpan model         |
| `app.py`            | Prototipe aplikasi web (Streamlit)                                                       |
| `requirements.txt`  | Daftar library                                                                           |

File yang dihasilkan: `komentar_mentah.csv`, `komentar_berlabel.csv`, `sampel_validasi.csv`, `hasil_perbandingan.csv`, `model_sentimen_mbg.joblib`, dan folder `grafik/`.

---

## 4. Instalasi

Disarankan memakai **Python resmi 3.11 atau 3.12** dari python.org (centang "Add python.exe to PATH" saat instalasi). Python bawaan MSYS2 tidak membawa pip dan sulit dipakai untuk library data science.

```
python -m pip install -r requirements.txt
```

Library utama: `pandas`, `scikit-learn`, `Sastrawi`, `matplotlib`, `seaborn`, `wordcloud`, `joblib`, `streamlit`, `google-api-python-client`.

---

## 5. Cara menjalankan

Jalankan perintah di folder proyek (folder yang berisi `latih.py`).

**Langkah 1. Siapkan API key YouTube (sekali saja)**

1. Buka console.cloud.google.com, buat project baru.
2. Di **APIs & Services -> Library**, aktifkan **YouTube Data API v3**.
3. Di **APIs & Services -> Credentials**, buat **API key** (disarankan dibatasi hanya ke YouTube Data API v3).

**Langkah 2. Ambil komentar**

Video ID adalah teks setelah `v=` pada URL. Untuk `youtube.com/watch?v=BmDTNa00Axo`, ID-nya `BmDTNa00Axo`.

PowerShell:

```
$env:YOUTUBE_API_KEY="KUNCI_KAMU"
python ambil_komentar.py BmDTNa00Axo
```

Command Prompt:

```
set YOUTUBE_API_KEY=KUNCI_KAMU
python ambil_komentar.py BmDTNa00Axo
```

Beberapa video: pisahkan ID dengan spasi. Hasil: `komentar_mentah.csv`.

**Langkah 3. EDA, praproses, dan label lexicon**

```
python siapkan_data.py
```

Hasil: `komentar_berlabel.csv`, `sampel_validasi.csv` (200 komentar), folder `grafik/`.

**Langkah 4. Validasi manual**
Buka `sampel_validasi.csv`, isi kolom `label_manual` dengan `positif`, `netral`, atau `negatif` (huruf kecil), simpan tetap sebagai CSV. Jangan menjalankan `siapkan_data.py` lagi sesudahnya, karena file ini akan tertimpa.

**Langkah 5. Latih dan evaluasi**

```
python latih.py
```

**Langkah 6. Jalankan prototipe**

```
python -m streamlit run app.py
```

---

## 6. Penjelasan tiap tahap

### 6.1 Pengambilan data

**Apa:** komentar diambil dari video YouTube bertopik MBG lewat YouTube Data API v3 (`commentThreads.list`, `maxResults=100`, dengan `pageToken` untuk halaman berikutnya). Setiap komentar disimpan bersama video_id, tanggal, dan jumlah like.

**Kenapa pakai API:** resmi, stabil, dan tidak melanggar ketentuan layanan seperti scraping halaman web. Konsekuensinya butuh API key dan ada batas kuota harian, yang cukup untuk tugas ini.

**Data proyek ini:** 1 video (ID `BmDTNa00Axo`), **1.127 komentar** terambil, **1.117** setelah komentar duplikat dibuang.

### 6.2 EDA (Exploratory Data Analysis)

**Apa:** mengenali data sebelum dipakai melatih model. Hasilnya menjadi dasar keputusan berikutnya.

| Analisis                     | Tujuan                             | Hasil proyek ini                                                                  |
| ---------------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| Cek data kosong dan duplikat | Menilai kualitas data              | Kosong 0, duplikat 0 (sudah dibuang saat pengambilan)                             |
| Panjang komentar             | Mengetahui karakter teks           | Mayoritas komentar pendek, sebagian besar di bawah 20 kata                        |
| Distribusi label             | Mendeteksi ketidakseimbangan kelas | Netral 700 (63,41%), negatif 330 (29,89%), positif 74 (6,70%) dari 1.104 komentar |
| Kata dominan (WordCloud)     | Melihat tema tiap kelas            | Lihat `grafik/wordcloud_*.png`                                                    |

**Kenapa penting:** distribusi label yang timpang menjadi alasan memakai `class_weight="balanced"` dan **macro-F1**, bukan akurasi saja.

**Catatan:** distribusi ini dihitung dari label **lexicon**. Kelas netral kemungkinan terlalu besar karena cakupan kamus terbatas (lihat bagian 6.4).

Setelah praproses, 13 komentar menjadi kosong (hanya berisi emoji, angka, atau tautan) dan dibuang, sehingga tersisa **1.104 komentar**.

### 6.3 Praproses teks

**Apa:** membersihkan dan menyeragamkan teks sebelum diubah menjadi angka. Komentar YouTube berantakan: huruf kapital acak, emoji, tautan, singkatan, kata tidak baku.

**Kenapa:** tanpa praproses, kata yang sama ("Bagus", "bagus", "bagusss") dianggap berbeda sehingga model menyebar informasinya ke banyak fitur.

| Tahap               | Proses                                                   | Contoh                                          |
| ------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| Case folding        | Semua huruf jadi kecil                                   | "MBG Bagus" -> "mbg bagus"                      |
| Data cleaning       | Hapus tautan, mention, hashtag, emoji, angka, tanda baca | "bagus!!! @budi" -> "bagus"                     |
| Normalisasi         | Kata tidak baku jadi baku                                | "bgt" -> "banget", "gak" -> "tidak"             |
| Tokenizing          | Pecah kalimat jadi daftar kata                           | "mbg tidak bagus" -> [mbg, tidak, bagus]        |
| Stopword removal    | Buang kata umum (yang, di, itu, sih), **kecuali negasi** | [yang, tidak, bagus] -> [tidak, bagus]          |
| Stemming (Sastrawi) | Ubah ke kata dasar                                       | "keracunan" -> "racun", "disebabkan" -> "sebab" |

**Kenapa kata negasi dipertahankan:** kata seperti "tidak", "bukan", "jangan", "belum", "kurang" membalik arah sentimen. "Tidak bagus" berarti kebalikan dari "bagus". Jika negasi dibuang, sentimen salah terbaca.

**Contoh nyata dari data:**

```
Mentah   : Keracunan anak sekolah yang disebabkan oleh Negara disebut biasa oleh orang Dungu 😳
Folding  : keracunan anak sekolah yang disebabkan oleh negara disebut biasa oleh orang dungu 😳
Cleaning : keracunan anak sekolah yang disebabkan oleh negara disebut biasa oleh orang dungu
Token    : [keracunan, anak, sekolah, yang, disebabkan, oleh, negara, disebut, biasa, oleh, orang, dungu]
Stopword : [keracunan, anak, sekolah, disebabkan, oleh, negara, disebut, biasa, oleh, orang, dungu]
Stemming : [racun, anak, sekolah, sebab, oleh, negara, sebut, biasa, oleh, orang, dungu]
```

### 6.4 Pelabelan lexicon

**Apa:** pemberian label awal (positif, netral, negatif) secara otomatis memakai kamus kata bersentimen.

**Rumus:**

```
skor = jumlah bobot semua kata bersentimen dalam komentar
       (bobot dibalik tandanya bila kata sebelumnya adalah negasi)

skor > 0  -> positif
skor < 0  -> negatif
skor = 0  -> netral
```

**Hitungan contoh:** pada komentar di atas, hanya "racun" yang ada di kamus (bobot -3), sehingga skor = -3 -> **negatif**. Pada "tidak setuju", kata "setuju" (bobot +2) didahului negasi sehingga dibalik menjadi -2.

**Kenapa pakai lexicon:** memberi label manual pada seluruh komentar terlalu lama. Lexicon memberi label awal dengan cepat, lalu sebagian diperiksa manual untuk mengukur kualitasnya.

**Kenapa kamus ikut di-stem:** komentar sudah melalui stemming, sehingga kata di kamus harus diubah ke bentuk dasar juga. Jika tidak, "setuju" di kamus tidak akan cocok dengan "tuju" di komentar. Kode `praproses.py` melakukannya otomatis lewat `LEXICON_STEM`.

**Kelemahan:** lexicon di proyek ini hanya **starter** yang kecil. Kata yang tidak ada di kamus bernilai 0 dan akhirnya jatuh ke kelas netral. Lexicon juga tidak membaca sarkasme. Untuk hasil lebih baik, gunakan lexicon InSet.

### 6.5 Validasi manual

**Apa:** 200 komentar sampel diberi label oleh manusia (kolom `label_manual` di `sampel_validasi.csv`), lalu dibandingkan dengan label lexicon.

**Fungsi:** (1) mengukur kualitas label otomatis, dan (2) memperbaiki label pada sampel itu. `latih.py` otomatis memakai label manual bila diisi.

**Panduan menilai:** positif = mendukung atau memuji MBG; negatif = mengkritik, menolak, menyindir, atau mengeluh; netral = bertanya atau menyatakan informasi tanpa sikap. Sindiran dinilai sesuai maksudnya.

**Hasil:** kesesuaian dengan lexicon hanya **42,00%** (84 dari 200). Artinya lexicon salah pada sekitar 58% sampel. Ini temuan penting dan alasan validasi manual diperlukan.

### 6.6 Pembagian data

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

- Data latih: 1.104 x 0,8 = **883**. Data uji: **221** (scikit-learn membulatkan jumlah data uji ke atas).
- `stratify=y` menjaga proporsi kelas di data latih dan uji sama dengan data asli. Tanpa ini, kelas positif yang sangat kecil bisa nyaris hilang dari salah satu bagian.
- `random_state=42` membuat pembagian bisa diulang dengan hasil yang sama.
- Data uji **tidak dipakai** saat pelatihan, supaya penilaian jujur.

### 6.7 TF-IDF

**Apa:** mengubah teks menjadi angka. SVM tidak bisa membaca kata, jadi tiap komentar diubah menjadi vektor, satu angka untuk tiap kata.

**Idenya:**

- **TF:** kata yang sering muncul di satu komentar dianggap penting untuk komentar itu.
- **IDF:** kata yang muncul di hampir semua komentar tidak membedakan, jadi bobotnya diturunkan; kata yang jarang dan khas diberi bobot tinggi.

**Rumus yang dipakai scikit-learn (smooth IDF):**

```
IDF(t)      = ln( (1 + N) / (1 + df(t)) ) + 1
TF-IDF(t,d) = TF(t,d) x IDF(t)
lalu tiap baris dinormalisasi (L2) sehingga panjang vektor = 1
```

N = jumlah komentar, df(t) = jumlah komentar yang memuat kata t. Versi sederhana log(N/df) memberi bobot 0 pada kata yang ada di semua komentar; versi yang dihaluskan memberi bobot 1 dan menghindari pembagian dengan nol.

**Contoh hitungan (4 komentar buatan, N = 4):**

| Komentar | Isi setelah praproses  |
| -------- | ---------------------- |
| 1        | mbg racun anak sekolah |
| 2        | mbg bagus manfaat      |
| 3        | mbg tidak tuju racun   |
| 4        | mbg gizi anak          |

| Kata        | df  | IDF                 |
| ----------- | --- | ------------------- |
| mbg         | 4   | ln(5/5) + 1 = 1,000 |
| anak, racun | 2   | ln(5/3) + 1 = 1,511 |
| sekolah     | 1   | ln(5/2) + 1 = 1,916 |

Komentar 1 (tiap kata muncul sekali): bobot mentah mbg 1,000, racun 1,511, anak 1,511, sekolah 1,916. Panjang vektor = akar(1,000² + 1,511² + 1,511² + 1,916²) = 3,039. Setelah normalisasi:

| Kata    | Bobot akhir |
| ------- | ----------- |
| mbg     | 0,329       |
| anak    | 0,497       |
| racun   | 0,497       |
| sekolah | 0,631       |

Kata khas ("sekolah") berbobot tertinggi, kata yang ada di semua komentar ("mbg") terendah.

**Pengaturan di proyek ini dan alasannya:**

| Pengaturan          | Nilai  | Alasan                                                                                                                                                                                                               |
| ------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_features`      | 5.000  | Hanya 5.000 kata atau pasangan kata terpopuler yang dipakai sebagai kolom. Ini batas atas; jumlah sebenarnya bisa lebih sedikit karena data kecil. Mengurangi noise dan beban komputasi. Mengikuti penelitian acuan. |
| `ngram_range`       | (1, 2) | Kata tunggal ("racun") dan pasangan kata ("tidak tuju"). Pasangan kata menangkap konteks seperti negasi.                                                                                                             |
| `min_df`            | 2      | Kata yang muncul di kurang dari 2 komentar dibuang karena terlalu langka (biasanya salah ketik atau noise).                                                                                                          |
| `max_df`            | 0,95   | Kata yang muncul di lebih dari 95% komentar dibuang karena terlalu umum.                                                                                                                                             |
| Di dalam `Pipeline` |        | TF-IDF hanya dipelajari dari data latih, sehingga kosakata data uji tidak bocor ke pelatihan.                                                                                                                        |

**Hasil:** matriks jarang (sparse), baris = komentar, kolom = fitur. Sebagian besar nilainya nol karena satu komentar hanya memuat sedikit kata.

Untuk melihat jumlah fitur sebenarnya, tambahkan di akhir `latih.py`:

```python
fitur = utama.named_steps["tfidf"].get_feature_names_out()
print("Jumlah fitur sebenarnya:", len(fitur))
```

### 6.8 Support Vector Machine (LinearSVC)

**Ide dasar:** SVM mencari batas pemisah (hyperplane) antar kelas, `w . x + b = 0`, dan memilih batas dengan **margin terlebar**. Titik data terdekat dengan batas disebut _support vector_.

```
Margin = 2 / ||w||
```

Margin besar berarti ||w|| kecil, dan batas yang lebar membuat model lebih tahan terhadap data baru.

**Soft margin:** data teks tidak bisa dipisahkan sempurna (ada ironi, kata ambigu, kalimat berpotongan), jadi SVM mengizinkan sedikit kesalahan lewat parameter **C**:

```
minimalkan   1/2 ||w||^2  +  C x (jumlah kesalahan)
```

C kecil = lebih toleran pada kesalahan, batas lebih longgar. C besar = ketat pada data latih, berisiko overfitting. Di proyek ini dipakai C = 1 (nilai bawaan). Catatan: LinearSVC di scikit-learn memakai _squared hinge loss_ untuk menghitung kesalahan, sedikit berbeda dari rumus buku.

**Tiga kelas (One-vs-Rest):** SVM aslinya memisahkan dua kelas. Untuk tiga kelas dilatih tiga penilai: positif lawan sisanya, netral lawan sisanya, negatif lawan sisanya. Tiap penilai memberi skor:

```
skor_k = w_k . x + b_k          prediksi = kelas dengan skor tertinggi (argmax)
```

Skor inilah yang tampil sebagai batang di grafik aplikasi. Kelas tidak harus berskor positif untuk terpilih; yang menentukan hanya skor mana yang tertinggi dibanding dua lainnya.

**Penanganan ketidakseimbangan (`class_weight="balanced"`):**

```
bobot_kelas = jumlah data latih / (jumlah kelas x jumlah data kelas itu)
```

Dengan perkiraan jumlah data latih per kelas (positif sekitar 52, netral sekitar 484, negatif sekitar 348, diturunkan dari jumlah data uji):

| Kelas   | Perkiraan bobot                |
| ------- | ------------------------------ |
| Positif | 883 / (3 x 52) = sekitar 5,66  |
| Netral  | 883 / (3 x 484) = sekitar 0,61 |
| Negatif | 883 / (3 x 348) = sekitar 0,85 |

Artinya salah memprediksi komentar positif "dihargai" sekitar 9 kali lebih mahal daripada salah memprediksi komentar netral. Model dipaksa memperhatikan kelas kecil dan tidak hanya menebak kelas besar.

**Kenapa SVM (LinearSVC) untuk proyek ini:**

1. Teks setelah TF-IDF berdimensi tinggi dan jarang; SVM linear bekerja baik pada kondisi itu.
2. Cepat dilatih pada data yang tidak terlalu besar (sekitar 1.100 komentar); model deep learning seperti IndoBERT butuh data dan komputasi lebih besar.
3. Mendukung penanganan kelas tidak seimbang lewat bobot kelas.
4. Sesuai dengan penelitian acuan, sehingga hasil bisa dibandingkan.

### 6.9 Model pembanding

Empat model dilatih pada data dan fitur TF-IDF yang sama:

| Model                          | Tujuan perbandingan                              |
| ------------------------------ | ------------------------------------------------ |
| LinearSVC (balanced)           | Model utama                                      |
| LinearSVC (tanpa bobot)        | Mengukur pengaruh `class_weight`                 |
| Naive Bayes (MultinomialNB)    | Pembanding sederhana yang umum dipakai pada teks |
| Logistic Regression (balanced) | Pembanding linear lain                           |

### 6.10 Evaluasi

**Metrik:**

```
Precision = TP / (TP + FP)       ketepatan prediksi satu kelas
Recall    = TP / (TP + FN)       kemampuan menemukan semua data satu kelas
F1        = 2 x P x R / (P + R)  keseimbangan precision dan recall
Accuracy  = benar / total
Macro-F1  = rata-rata F1 semua kelas (tiap kelas setara)
```

**Kenapa macro-F1, bukan accuracy saja:** accuracy dihitung dari semua komentar, sehingga didominasi kelas besar. Model yang hanya pandai di netral dan negatif tetap tampak akurat. Macro-F1 memperlakukan semua kelas setara, sehingga kelas positif yang lemah ikut terlihat.

**Hitungan dari hasil LinearSVC (data uji = 221):**

| Kelas   | Benar | Aktual | Diprediksi | Recall           | Precision        | F1     |
| ------- | ----- | ------ | ---------- | ---------------- | ---------------- | ------ |
| Positif | 5     | 13     | 12         | 5/13 = 0,3846    | 5/12 = 0,4167    | 0,4000 |
| Netral  | 101   | 121    | 128        | 101/121 = 0,8347 | 101/128 = 0,7891 | 0,8112 |
| Negatif | 63    | 87     | 81         | 63/87 = 0,7241   | 63/81 = 0,7778   | 0,7500 |

```
Accuracy = (5 + 101 + 63) / 221 = 169 / 221 = 0,7647
Macro-F1 = (0,4000 + 0,8112 + 0,7500) / 3 = 0,6537
```

Jumlah "benar" dan "diprediksi" diturunkan dari rasio pada output; matriks lengkap ada di `grafik/confusion_matrix.png`.

**Cross-validation 5-fold (`macro_F1_CV5`):** data dibagi 5 bagian stratified; model dilatih 5 kali dengan tiap bagian bergantian menjadi data uji; macro-F1 yang dilaporkan adalah rata-rata kelima hasil. Ini lebih stabil daripada satu kali pembagian, terutama untuk data kecil.

### 6.11 Prototipe (Streamlit)

`app.py` memuat model sekali (`@st.cache_resource`), menerima satu komentar, menjalankan praproses yang sama seperti saat pelatihan, lalu menampilkan:

- label sentimen (positif, netral, atau negatif),
- teks setelah praproses,
- grafik batang skor tiap kelas.

**Cara membaca grafik skor:** batang tertinggi adalah prediksi. Selisih antar batang menunjukkan keyakinan: jika batang tertinggi jauh di atas yang lain, model yakin; jika semua batang mepet atau negatif, model ragu. Skor bukan persentase.

---

## 7. Hasil

**Data:** 1 video YouTube, 1.127 komentar diambil, 1.117 tanpa duplikat, **1.104** setelah praproses. Label lexicon: netral 700 (63,41%), negatif 330 (29,89%), positif 74 (6,70%). Kesesuaian lexicon dengan label manual: **42%**. Data latih 883, data uji 221.

**Perbandingan model:**

| Model                   | Accuracy | Macro-F1 | Macro-F1 CV5 |
| ----------------------- | -------- | -------- | ------------ |
| LinearSVC (balanced)    | 0,7647   | 0,6537   | 0,6573       |
| LinearSVC (tanpa bobot) | 0,7738   | 0,6866   | 0,6288       |
| Naive Bayes             | 0,7014   | 0,4670   | 0,4892       |
| Logistic Regression     | 0,7873   | 0,6997   | 0,6666       |

**Laporan klasifikasi LinearSVC (balanced):**

| Kelas     | Precision | Recall | F1     | Support |
| --------- | --------- | ------ | ------ | ------- |
| Positif   | 0,4167    | 0,3846 | 0,4000 | 13      |
| Netral    | 0,7891    | 0,8347 | 0,8112 | 121     |
| Negatif   | 0,7778    | 0,7241 | 0,7500 | 87      |
| Accuracy  |           |        | 0,7647 | 221     |
| Macro avg | 0,6612    | 0,6478 | 0,6537 | 221     |

---

## 8. Kenapa hasilnya begini

- **Accuracy 76% tetapi macro-F1 65%:** accuracy didominasi kelas netral dan negatif, sedangkan kelas positif (F1 0,40) menarik rata-rata macro ke bawah.
- **Kelas positif paling lemah:** datanya paling sedikit (13 sampel di data uji, sekitar 6-7% dari seluruh data), kosakata positifnya terbatas, dan banyak komentar "positif" berupa sindiran. Karena hanya 13 sampel, satu komentar salah menggeser recall sekitar 8%, sehingga angka positif tidak stabil.
- **Naive Bayes paling buruk:** tanpa penanganan kelas kecil, model cenderung menebak kelas besar.
- **SVM sebanding dengan Logistic Regression:** keduanya model linear pada fitur yang sama. Selisihnya kecil (sekitar 0,01-0,05) dan data uji hanya 221 komentar, sehingga tidak bisa disimpulkan SVM lebih unggul.
- **`class_weight` membantu pada cross-validation:** macro-F1 CV naik dari 0,6288 menjadi 0,6573. Pada satu kali pembagian, hasilnya justru turun, jadi buktinya campuran; cross-validation lebih dapat dipercaya.
- **Macro-F1 CV (0,6573) hampir sama dengan macro-F1 data uji (0,6537):** hasilnya konsisten.
- **Contoh "mbg jelek" diprediksi netral:** jika kata "jelek" jarang muncul di data latih, kata itu dibuang oleh `min_df=2`, dan yang tersisa hanya "mbg" yang paling sering muncul di komentar netral. Model hanya sebaik data latihnya.

---

## 9. Keterbatasan

1. **Kualitas label.** Label manual baru 200 dari 1.104 komentar (sekitar 18%). Sisanya memakai label lexicon yang hanya 42% sesuai dengan penilaian manusia. Model sebagian dilatih dan diuji pada label berisik, sehingga angka evaluasi mengukur kecocokan dengan label campuran, bukan kebenaran sentimen sepenuhnya.
2. **Data satu video.** Hasil tidak mewakili seluruh opini publik tentang MBG atau seluruh percakapan di platform lain.
3. **Kelas positif sangat sedikit,** sehingga metrik kelas positif tidak stabil.
4. **Lexicon kecil dan tidak membaca sarkasme atau ironi.**
5. **Data uji kecil (221),** sehingga selisih kecil antar model tidak bermakna.

---

## 10. Saran pengembangan

- Tambah data dari beberapa video dan kanal.
- Perbanyak label manual dan libatkan lebih dari satu penilai (hitung kesepakatan antar penilai, misalnya Cohen's kappa).
- Gunakan lexicon yang lebih lengkap (misalnya InSet) atau tambahkan kata dari data sendiri.
- Seimbangkan kelas positif (pengumpulan data tambahan atau resampling).
- Coba model berbasis bahasa Indonesia seperti IndoBERT.
- Tuning parameter C dengan grid search dan cross-validation.

---

## 11. Deploy

**Opsi 1, Streamlit Community Cloud (paling mudah, memakai `app.py` apa adanya):**

1. Siapkan folder berisi `app.py`, `praproses.py`, `model_sentimen_mbg.joblib`, dan `requirements.txt` ringan (`streamlit`, `scikit-learn==VERSI_KAMU`, `Sastrawi`, `joblib`). Versi scikit-learn harus sama dengan saat melatih (`python -m pip show scikit-learn`).
2. Unggah ke repository GitHub (publik).
3. Buka share.streamlit.io, login dengan GitHub, **New app**, pilih repository, branch, dan file utama `app.py`. Pilih versi Python yang sama dengan laptop, lalu **Deploy**.

**Opsi 2, Vercel:** Streamlit tidak berjalan di Vercel karena Vercel menjalankan fungsi serverless. Gunakan versi FastAPI + HTML (endpoint `/predict` dan halaman `index.html`).

**Keamanan:** jangan mengunggah `ambil_komentar.py` yang berisi API key ke repository publik. Gunakan environment variable dan hapus key dari file sebelum dibagikan. Jika key pernah terlihat di tempat umum, hapus dan buat key baru di Credentials.

---

## 12. Pemecahan masalah

| Masalah                                                    | Penyebab dan solusi                                                                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `No module named pip`                                      | VS Code memakai Python bawaan MSYS2. Pasang Python resmi, lalu pilih interpreter dengan `Ctrl+Shift+P` -> **Python: Select Interpreter**. |
| `Import "..." could not be resolved`                       | Interpreter VS Code berbeda dengan yang dipakai `pip`. Pilih interpreter yang benar dan jalankan **Developer: Reload Window**.            |
| `set YOUTUBE_API_KEY=...` tidak jalan                      | Itu sintaks Command Prompt. Di PowerShell pakai `$env:YOUTUBE_API_KEY="KUNCI"`.                                                           |
| Skrip langsung berhenti tanpa hasil                        | Skrip dijalankan tanpa VIDEO_ID (misalnya lewat tombol Run). Jalankan lewat terminal dengan ID video di belakang perintah.                |
| `commentsDisabled`                                         | Video menutup kolom komentar. Pilih video lain.                                                                                           |
| `FileNotFoundError: komentar_berlabel.csv`                 | `siapkan_data.py` belum dijalankan. Jalankan dulu sebelum `latih.py`.                                                                     |
| Isian `label_manual` hilang                                | `siapkan_data.py` dijalankan ulang dan menimpa `sampel_validasi.csv`. Simpan salinan file sebelum menjalankan ulang.                      |
| `app.py` error saat memuat model                           | `model_sentimen_mbg.joblib` belum ada (jalankan `latih.py`) atau versi scikit-learn berbeda dari saat melatih.                            |
| Prediksi tidak sesuai harapan                              | Kata di luar kosakata data latih diabaikan. Tambah data, atau coba kalimat lebih panjang yang memuat kata yang sering muncul di data.     |
| Hasil lexicon terlihat salah (mis. "setuju" tidak terbaca) | Pastikan `praproses.py` versi terbaru yang memakai `LEXICON_STEM`, lalu jalankan ulang `siapkan_data.py`.                                 |

---

## 13. Referensi

1. Andianto, B., & Februariyanti, H. (2026). Analisis Opini Publik Terhadap Program Makan Bergizi Gratis (MBG) pada Media Sosial Youtube Menggunakan Algoritma Support Vector Machine. _Jurnal Teknik Informatika dan Sistem Informasi (Jatisi)_, 13(2), 111-124.
2. Najib, L., Mahfudh, A. A., & Bakhri, S. (2026). Analisis Sentimen Persepsi Publik Terhadap Program MBG pada Komentar YouTube Menggunakan Naive Bayes dan Resampling. _Technology and Science (BITS)_, 7(4). https://doi.org/10.47065/bits.v7i4.9400
3. Majidah, S. H., & Hendikawati, P. (2026). Analisis Sentimen Program Makan Bergizi Gratis Menggunakan SVM dan KNN dengan TF-IDF dan TF-ABS. _E-Jurnal Matematika_, 15(1), 1-9. https://doi.org/10.24843/MTK.2026.v15.i01.p496
4. Nugroho, D. C., Agustina, A. D., Maulana, B., Darussalam, F., Asyraf, M., & Lubis, B. O. (2025). Penerapan Metode Support Vector Machine untuk Analisis Sentimen Publik pada Program Makan Bergizi Gratis. _INFOTECH Journal_, 11(2), 474-479.
5. Koto, F., & Rahmaningtyas, G. Y. (2017). InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs. _International Conference on Asian Language Processing (IALP)_.
6. Dokumentasi scikit-learn: `TfidfVectorizer`, `LinearSVC`, `classification_report`, `StratifiedKFold`.
