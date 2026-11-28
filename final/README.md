# Task 2 — Tim Perintis (Datavidia 11.0)

Klasifikasi fenomena dari pasangan citra satelit `pre` / `post` ke 6 kelas: `Earthquake`, `Explosion`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`. Tidak ada label per foto. Cluster dibentuk model, lalu nama kelas ditetapkan **per cluster** lewat pengamatan montase oleh AI (Claude), tanpa tinjauan tim. Kamus nama cluster ditulis eksplisit di `01_model.ipynb` (bagian 2). Nama cluster tidak disetel dari skor leaderboard. Prediksi akhir dihasilkan classifier.

## Isi folder

```
final/
├── 01_model.ipynb        # citra → fitur → muat cluster B + kamus nama → label spreading → classifier → model/
├── 02_submission.ipynb   # citra → fitur → muat model/ → submission/submission_final.csv (tanpa .fit())
├── events.parquet        # hasil B (Fahmi v3-7): Id, event_id, kedekatan, kelas_cluster, yakin_cluster
├── sub_events.parquet    # pemetaan foto -> induk/sub-cluster (untuk cek hash)
├── cluster_names.json    # kelas, yakin, alasan, penilai per cluster
├── pca.joblib, mean_x.npy  # ruang Z milik B (dimuat, tidak di-fit ulang)
├── model/                # dibuat oleh 01_model.ipynb
├── requirements.txt      # ditulis otomatis oleh 01_model.ipynb
└── README.md
```

Bobot DINOv2-L untuk jalan offline (±1,2 GB) disimpan di Drive, bukan di folder ini. Taruh di `weights/dinov2_large/` (dicari otomatis), atau isi `WEIGHTS_DIR` di sel Pengaturan.

## Cara menjalankan

1. Taruh dataset relatif terhadap folder ini: `Dataset/Fenomena/<Id>/pre.jpg`, `Dataset/Fenomena/<Id>/post.jpg`, dan `Dataset/sample_submission.csv`. Kalau letaknya beda, ubah `DATA_DIR` dan `SAMPLE_SUB` di sel Pengaturan kedua notebook.
2. `pip install -r requirements.txt`
3. Jalankan `01_model.ipynb` sampai selesai. Hasilnya mengisi `model/`.
4. Jalankan `02_submission.ipynb`. Hasilnya `submission/submission_final.csv`. Sel terakhir mengecek prediksi sama 100% dengan `model/pred_train.csv`.

Ekstraksi DINOv2-L (FP32, TTA 8 arah, 6.360 gambar) butuh GPU. Dijalankan di Kaggle Tesla T4 (1 GPU).

## Metode

| Bagian | Isi |
|---|---|
| A — Pengolah gambar (Hafizh) | Preprocessing (mask area hitam, shared crop, normalisasi warna selektif), embedding DINOv2-L (`vit_large_patch14_dinov2.lvd142m`, Apache-2.0, 308 px) dengan TTA D4 8 arah, FP32 deterministik. |
| B — Cluster & nama kelas (Fahmi) | Embedding pre+post → L2 per blok → PCA 64 → dibulatkan 4 desimal → Spectral Clustering K=40 (kNN 15, seed 42). 4 cluster campur (2, 10, 29, 39) dipecah dengan Spectral Clustering menjadi sub-cluster 40–49, total 46 cluster. Nama kelas per cluster dari pengamatan montase oleh AI (Claude), tanpa tinjauan tim. |
| C — Model akhir (Seno) | Seed = foto dengan kedekatan ke pusat cluster ≥ 0,5 → label spreading (kNN 15, alpha 0,2) → StandardScaler → PCA 128 → Logistic Regression (`class_weight="balanced"`, C=1,0), bobot sampel = kedekatan × (1,0 yakin / 0,5 ragu). Fitur hanya embedding per foto, tanpa `event_id`. |

Seed 42 di semua langkah. Urutan Id mengikuti `sample_submission.csv`.

**Keterbatasan:** tidak ada cluster yang dinamai `Explosion`, jadi kelas itu tidak pernah diprediksi (Macro F1 maksimal ±0,83). 27 dari 46 cluster berstatus `ragu`.

## Checklist sebelum dikumpulkan (hapus bagian ini setelah selesai)

- [x] Fungsi A yang dikunci ditempel di sel bagian 1 **kedua** notebook
- [x] `MODE_UJI = False` di kedua notebook
- [ ] `01_model.ipynb` dijalankan ulang → `model/` baru, `requirements.txt` tertulis otomatis
- [ ] `02_submission.ipynb` lulus cek 100%
- [ ] Path dataset sesuai aturan pengumpulan
- [ ] Isi `final/` + bobot diunggah ke Drive, link "siapa pun yang punya link bisa melihat", dicek di mode incognito
