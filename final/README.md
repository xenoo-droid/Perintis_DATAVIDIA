# Task 2 — Tim Perintis (Datavidia 11.0)

Klasifikasi fenomena dari pasangan citra satelit `pre` / `post` ke 6 kelas: `Earthquake`, `Explosion`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`. Tidak ada label manusia per foto maupun per kelompok. Nama kelas hanya ditentukan oleh aturan kode di `rules.json`.

## Isi folder

```
final/
├── 01_model.ipynb        # citra → fitur → cluster → penamaan → label spreading → classifier → model/
├── 02_submission.ipynb   # citra → fitur → muat model/ → submission/submission_final.csv (tanpa .fit())
├── rules.json            # aturan penamaan kelas (dikunci Fahmi, 2026-11-27 18:28)
├── model/                # dibuat oleh 01_model.ipynb
├── requirements.txt      # ditulis otomatis oleh 01_model.ipynb
└── README.md
```

Bobot DINOv2 untuk jalan offline (±1,2 GB) disimpan di Drive, bukan di folder ini.

## Cara menjalankan

1. Taruh dataset relatif terhadap folder ini: `Dataset/Fenomena/<Id>/pre.jpg`, `Dataset/Fenomena/<Id>/post.jpg`, dan `Dataset/sample_submission.csv`. Kalau letaknya beda, ubah `DATA_DIR` dan `SAMPLE_SUB` di sel Pengaturan kedua notebook.
2. `pip install -r requirements.txt`
3. Jalankan `01_model.ipynb` sampai selesai. Hasilnya mengisi `model/`.
4. Jalankan `02_submission.ipynb`. Hasilnya `submission/submission_final.csv`. Sel terakhir mengecek prediksi sama 100% dengan `model/pred_train.csv`.

## Metode

| Bagian | Isi |
|---|---|
| A — Pengolah gambar (Hafizh) | Preprocessing (mask area hitam, shared crop, normalisasi warna, cabang grayscale + CLAHE), embedding DINOv2-L dengan TTA 8 arah, indikator fisik `phys_v4`, statistik patch perubahan. |
| B — Cluster & nama kelas (Fahmi) | Embedding pre+post → PCA 64 → Spectral Clustering K=20 → median indikator per kejadian → skor 6 kelas dari `rules.json` → nama kelas per kejadian. |
| C — Model akhir (Seno) | Seed = 50% foto terdekat pusat setiap cluster → label spreading (kNN 15) → Logistic Regression (`class_weight="balanced"`) dari embedding per foto saja, tanpa `event_id`. |

Seed 42 di semua langkah. Urutan Id mengikuti `sample_submission.csv`.

## Checklist sebelum dikumpulkan (hapus bagian ini setelah selesai)

- [ ] Fungsi A yang dikunci ditempel di sel bagian 1 **kedua** notebook
- [ ] `MODE_UJI = False` di kedua notebook
- [ ] `01_model.ipynb` dijalankan ulang → `model/` baru, `requirements.txt` tertulis otomatis
- [ ] `02_submission.ipynb` lulus cek 100%
- [ ] Path dataset sesuai aturan pengumpulan
- [ ] Isi `final/` + bobot diunggah ke Drive, link "siapa pun yang punya link bisa melihat", dicek di mode incognito
