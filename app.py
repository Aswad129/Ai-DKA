"""Prototipe: jalankan dengan  streamlit run app.py"""
import joblib
import streamlit as st
from praproses import praproses

st.set_page_config(page_title="Sentimen MBG", page_icon="")
st.title("Klasifikasi Sentimen Komentar MBG")
st.caption("TF-IDF + Support Vector Machine (LinearSVC)")


@st.cache_resource
def muat():
    return joblib.load("model_sentimen_mbg.joblib")


model = muat()
teks = st.text_area("Masukkan komentar:", height=120)
if st.button("Prediksi") and teks.strip():
    bersih = praproses(teks)
    hasil = model.predict([bersih])[0]
    skor = model.decision_function([bersih])[0]
    ikon = {"positif": "🟢", "netral": "⚪", "negatif": "🔴"}[hasil]
    st.subheader(f"{ikon} {hasil.upper()}")
    st.write("Teks setelah praproses:", f"`{bersih}`")
    st.bar_chart(dict(zip(model.classes_, skor)))
