"""Tahap 2 - EDA, praproses, dan pelabelan lexicon.

Masukan : komentar_mentah.csv (kolom: komentar)
Keluaran: komentar_berlabel.csv, folder grafik/
Label lexicon hanya label AWAL. Validasi manual: buka file 'sampel_validasi.csv',
isi kolom label_manual, lalu jalankan latih.py (otomatis memakai label manual jika ada).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from praproses import praproses, label_lexicon, skor_lexicon

os.makedirs("grafik", exist_ok=True)
df = pd.read_csv("komentar_mentah.csv")

# --- EDA data mentah ---
print("Baris awal:", len(df), "| kosong:", df["komentar"].isna().sum(),
      "| duplikat:", df["komentar"].duplicated().sum())
df = df.dropna(subset=["komentar"]).drop_duplicates(subset="komentar")
df["jumlah_kata"] = df["komentar"].str.split().str.len()
plt.figure(figsize=(6, 4)); df["jumlah_kata"].clip(upper=100).hist(bins=30)
plt.title("Distribusi panjang komentar (kata)"); plt.xlabel("jumlah kata")
plt.tight_layout(); plt.savefig("grafik/panjang_komentar.png", dpi=150); plt.close()

# --- Praproses + label ---
df["teks_bersih"] = df["komentar"].apply(praproses)
df = df[df["teks_bersih"].str.strip() != ""].copy()
df["skor"] = df["teks_bersih"].apply(skor_lexicon)
df["label"] = df["teks_bersih"].apply(label_lexicon)

# --- EDA label ---
dist = df["label"].value_counts()
print("\nDistribusi label:\n", dist, "\n", (dist / len(df) * 100).round(2))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
dist.plot.bar(ax=ax[0], color=["#d62728", "#2ca02c", "#7f7f7f"][: len(dist)])
ax[0].set_title("Jumlah komentar per sentimen")
dist.plot.pie(ax=ax[1], autopct="%.1f%%"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("grafik/distribusi_sentimen.png", dpi=150); plt.close()

for kelas in dist.index:
    teks = " ".join(df.loc[df["label"] == kelas, "teks_bersih"])
    if teks.strip():
        WordCloud(width=800, height=400, background_color="white").generate(teks) \
            .to_file(f"grafik/wordcloud_{kelas}.png")

df.to_csv("komentar_berlabel.csv", index=False, encoding="utf-8-sig")

# --- Sampel untuk validasi manual (isi kolom label_manual) ---
n = min(200, len(df))
sampel = df.sample(n, random_state=42)[["komentar", "teks_bersih", "label"]].copy()
sampel["label_manual"] = ""
sampel.to_csv("sampel_validasi.csv", index=False, encoding="utf-8-sig")
print(f"\nTersimpan: komentar_berlabel.csv ({len(df)} baris), sampel_validasi.csv ({n} baris), grafik/")
