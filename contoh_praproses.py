import pandas as pd
from praproses import bersihkan, tokenisasi, hapus_stopword, stemming

df = pd.read_csv("komentar_mentah.csv")
for k in df["komentar"].dropna().sample(3, random_state=1):
    print("Mentah   :", k)
    print("Folding  :", k.lower())
    b = bersihkan(k)
    print("Cleaning :", b)
    t = tokenisasi(b)
    print("Token    :", t)
    s = hapus_stopword(t)
    print("Stopword :", s)
    print("Stemming :", stemming(s))
    print()