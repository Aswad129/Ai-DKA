"""Praproses teks dan pelabelan lexicon (dipakai bersama oleh semua skrip)."""
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_stemmer = StemmerFactory().create_stemmer()

# Kata negasi SENGAJA tidak dihapus karena membalik arah sentimen.
NEGASI = {"tidak", "bukan", "jangan", "belum", "kurang", "tak", "gak", "nggak", "ga"}

STOPWORD = {
    "yang", "di", "ke", "dari", "dan", "ini", "itu", "untuk", "dengan", "pada",
    "adalah", "atau", "juga", "ada", "saya", "kita", "kami", "dia", "mereka",
    "nya", "sih", "dong", "deh", "lah", "kan", "ya", "aja", "saja", "si", "para",
}

# Normalisasi kata tidak baku -> baku (tambahkan sesuai data kalian)
NORMALISASI = {
    "gak": "tidak", "nggak": "tidak", "ga": "tidak", "tak": "tidak",
    "bgt": "banget", "yg": "yang", "dgn": "dengan", "utk": "untuk",
    "krn": "karena", "tdk": "tidak", "sy": "saya", "udh": "sudah", "udah": "sudah",
}

# Lexicon STARTER (kecil). Ganti/perluas dengan lexicon InSet (Koto & Rahmaningtyas, 2017).
# Kata dalam bentuk dasar (setelah stemming). Format: kata -> bobot
LEXICON = {
    # positif
    "bagus": 2, "manfaat": 2, "bantu": 2, "sehat": 2, "gizi": 1, "dukung": 2,
    "setuju": 2, "baik": 2, "mantap": 2, "bagus": 2, "terima": 1, "kasih": 1,
    "senang": 2, "cerdas": 1, "sukses": 2, "hebat": 2, "solusi": 1, "peduli": 2,
    # negatif
    "korupsi": -3, "racun": -3, "keracunan": -3, "buruk": -2, "gagal": -2,
    "bohong": -3, "rugi": -2, "mahal": -1, "tolak": -2, "bubar": -2, "busuk": -2,
    "kritik": -1, "beban": -2, "salah": -2, "jelek": -2, "parah": -2, "khawatir": -1,
    "hutang": -2, "utang": -2, "tipu": -3, "bahaya": -2, "basi": -2, "mubazir": -2,
}


def bersihkan(teks: str) -> str:
    """Case folding + data cleaning + normalisasi."""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)      # tautan
    teks = re.sub(r"[@#]\w+", " ", teks)                # mention / hashtag
    teks = re.sub(r"[^a-z\s]", " ", teks)               # angka, tanda baca, emoji
    teks = re.sub(r"\s+", " ", teks).strip()
    return " ".join(NORMALISASI.get(k, k) for k in teks.split())


def tokenisasi(teks: str) -> list:
    return teks.split()


def hapus_stopword(token: list) -> list:
    return [t for t in token if t not in STOPWORD or t in NEGASI]


def stemming(token: list) -> list:
    return [t if t in NEGASI else _stemmer.stem(t) for t in token]


def praproses(teks: str) -> str:
    """Pipeline lengkap: teks mentah -> teks bersih (siap TF-IDF)."""
    token = tokenisasi(bersihkan(teks))
    token = hapus_stopword(token)
    token = stemming(token)
    return " ".join(token)


def skor_lexicon(teks_bersih: str) -> int:
    """Skor sentimen; kata negasi sebelum kata bersentimen membalik tandanya."""
    token = teks_bersih.split()
    skor = 0
    for i, t in enumerate(token):
        if t in LEXICON:
            bobot = LEXICON[t]
            if i > 0 and token[i - 1] in NEGASI:
                bobot = -bobot
            skor += bobot
    return skor


def label_lexicon(teks_bersih: str) -> str:
    s = skor_lexicon(teks_bersih)
    return "positif" if s > 0 else "negatif" if s < 0 else "netral"
