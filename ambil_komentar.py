"""Tahap 1 - Ambil komentar YouTube via YouTube Data API v3.

Pemakaian:
    set YOUTUBE_API_KEY=xxxx            (Windows)  /  export YOUTUBE_API_KEY=xxxx
    python ambil_komentar.py VIDEO_ID1 VIDEO_ID2 ...
VIDEO_ID = bagian setelah ?v= pada URL video.
"""
import os
import sys
import pandas as pd
from googleapiclient.discovery import build


def ambil(video_id: str, youtube) -> list:
    hasil, token = [], None
    while True:
        resp = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            pageToken=token, textFormat="plainText",
        ).execute()
        for item in resp["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            hasil.append({"video_id": video_id, "komentar": s["textDisplay"],
                          "tanggal": s["publishedAt"], "like": s["likeCount"]})
        token = resp.get("nextPageToken")
        if not token:
            return hasil


if __name__ == "__main__":
    kunci = "AIzaSyBoHBBMdYIxe0nNwUfiRkso8-VpLXMYc7c"
    daftar_video = ["BmDTNa00Axo"]      # tambah ID lain dipisah koma

    yt = build("youtube", "v3", developerKey=kunci)
    semua = []
    for vid in daftar_video:
        data = ambil(vid, yt)
        print(f"{vid}: {len(data)} komentar")
        semua += data
    df = pd.DataFrame(semua).drop_duplicates(subset="komentar")
    df.to_csv("komentar_mentah.csv", index=False, encoding="utf-8-sig")
    print(f"Tersimpan: komentar_mentah.csv ({len(df)} baris)")