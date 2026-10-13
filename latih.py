"""Tahap 3 - Latih & evaluasi TF-IDF + LinearSVC, bandingkan dengan baseline."""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

os.makedirs("grafik", exist_ok=True)
df = pd.read_csv("komentar_berlabel.csv")

# Pakai label manual (hasil validasi) bila sudah diisi
if os.path.exists("sampel_validasi.csv"):
    v = pd.read_csv("sampel_validasi.csv").dropna(subset=["label_manual"])
    v = v[v["label_manual"].isin(["positif", "netral", "negatif"])]
    if len(v):
        sama = (v["label"] == v["label_manual"]).mean()
        print(f"Validasi manual: {len(v)} komentar, kesesuaian dgn lexicon = {sama:.2%}")
        peta = dict(zip(v["komentar"], v["label_manual"]))
        df["label"] = df.apply(lambda r: peta.get(r["komentar"], r["label"]), axis=1)

X, y = df["teks_bersih"].fillna(""), df["label"]
KELAS = ["positif", "netral", "negatif"]
X_lat, X_uji, y_lat, y_uji = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Data latih: {len(X_lat)} | Data uji: {len(X_uji)}")


def buat(clf):
    return Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),min_df=2, max_df=0.95)), ("clf", clf)])


kandidat = {
    "LinearSVC (balanced)": buat(LinearSVC(class_weight="balanced", random_state=42)),
    "LinearSVC (tanpa bobot)": buat(LinearSVC(random_state=42)),
    "Naive Bayes": buat(MultinomialNB()),
    "Logistic Regression": buat(LogisticRegression(max_iter=1000, class_weight="balanced")),
}

skf = StratifiedKFold(5, shuffle=True, random_state=42)
baris = []
for nama, m in kandidat.items():
    m.fit(X_lat, y_lat)
    p = m.predict(X_uji)
    cv = cross_val_score(m, X, y, cv=skf, scoring="f1_macro").mean()
    baris.append({"model": nama, "accuracy": accuracy_score(y_uji, p),
                  "macro_F1": f1_score(y_uji, p, average="macro"), "macro_F1_CV5": cv})
perbandingan = pd.DataFrame(baris).round(4)
print("\n=== Perbandingan model ===\n", perbandingan.to_string(index=False))
perbandingan.to_csv("hasil_perbandingan.csv", index=False)

# --- Model utama: LinearSVC balanced ---
utama = kandidat["LinearSVC (balanced)"]
pred = utama.predict(X_uji)
print("\n=== Laporan klasifikasi (model utama) ===")
print(classification_report(y_uji, pred, labels=KELAS, digits=4, zero_division=0))

cm = confusion_matrix(y_uji, pred, labels=KELAS)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=KELAS, yticklabels=KELAS)
plt.xlabel("Label prediksi"); plt.ylabel("Label aktual"); plt.title("Confusion Matrix - LinearSVC")
plt.tight_layout(); plt.savefig("grafik/confusion_matrix.png", dpi=150); plt.close()

joblib.dump(utama, "model_sentimen_mbg.joblib")
print("Model tersimpan: model_sentimen_mbg.joblib")
