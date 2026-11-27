# Aturan Repo GitHub — Task 2 (Tim Perintis)

Kode dan output disimpan di GitHub. Dataset disimpan lokal di masing-masing laptop. Folder yang siap dikumpulkan diunggah ke Drive, lalu link-nya dikumpulkan.

| Peran | Orang | Menjalankan di |
|---|---|---|
| A — Pengolah gambar | Hafizh | Kaggle |
| B — Cluster & nama kelas | Fahmi | Lokal |
| C — Model akhir & submission | Seno | Kaggle |

---

## 1. Struktur

```
Task-2/
├── README.md
├── .gitignore
├── A_hafizh/
│   ├── A_fitur.ipynb          # satu notebook Hafizh, terus diperbarui
│   ├── v2/                    # output run v2 + meta.json
│   └── v3/
├── B_fahmi/
│   ├── B_kejadian.ipynb       # satu notebook Fahmi
│   ├── v1/
│   └── v2/
├── C_seno/
│   ├── C_model.ipynb          # satu notebook Seno
│   ├── run1/
│   └── run2/
├── final/
│   ├── 01_model.ipynb
│   ├── 02_submission.ipynb
│   └── model/
└── Dataset/                   # lokal, tidak di-commit
```

- **Satu notebook per orang**, diperbarui terus. Riwayat kode disimpan lewat commit.
- **Output setiap run** disimpan di subfolder versi baru (`v1`, `v2`, … / `run1`, `run2`, …).
- **Dataset** ditaruh di `Task-2/Dataset/` di laptop masing-masing.

---

## 2. Alur Setiap Kali Run

1. Ubah notebook milik sendiri.
2. Di sel Pengaturan, set `OUTPUT_DIR` ke folder versi baru.
3. Jalankan notebook.
4. Isi `meta.json` di folder versi itu.
5. **Commit notebook + folder output bersamaan**, lalu push.

**Khusus Kaggle (Hafizh & Seno):** jalankan dengan **Save Version**, unduh output dan notebook-nya, taruh di folder masing-masing, lalu commit.

---

## 3. Isi `meta.json`

```json
{
  "versi": "v2",
  "oleh": "Fahmi",
  "tanggal": "2026-11-27 21:30",
  "input": {"fitur": "A_hafizh/v2"},
  "pengaturan": {"metode": "spectral", "K": 20, "pca": 64, "seed": 42},
  "catatan": "pre+post, kNN 15"
}
```

---

## 4. Format Pesan Commit

```
[A] v3: DINOv3 SAT 256px
[B] v2: spectral K20, PCA 64
[C] run1: LogReg, submit 1
```

Untuk melihat kode yang menghasilkan suatu versi, buka commit dengan pesan versi tersebut di GitHub.

---

## 5. Aturan Main

1. Hanya ubah **notebook dan folder milik sendiri**.
2. **Jangan menimpa** folder versi lama. Setiap run masuk ke folder versi baru.
3. **Commit notebook dan output bersamaan** setiap selesai run. Kalau notebook diubah lagi sebelum commit, kode versi sebelumnya hilang dari riwayat.
4. `git pull` sebelum `git push`.
5. Simpan notebook **beserta output sel-selnya** (jangan di-clear).
6. Seed `42`, urutan Id = `sample_submission.csv`, nama kelas persis: `Earthquake`, `Explosion`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`.

---

## 6. `.gitignore`

```
Dataset/
weights/
*.pt
*.pth
*.safetensors
.ipynb_checkpoints/
__pycache__/
```

GitHub menolak file di atas 100 MB. File `.npy` embedding (±13 MB) dan `.parquet` aman. Bobot DINOv2/DINOv3 (±1,2 GB) disimpan di Drive, bukan GitHub.

---

## 7. Sel Pengaturan (contoh)

```python
import os
from pathlib import Path

DI_KAGGLE = os.path.exists("/kaggle/input")

DATA_DIR   = Path("/kaggle/input/NAMA-DATASET") if DI_KAGGLE else Path("../Dataset")
INPUT_DIR  = Path("../A_hafizh/v2")       # versi input dari orang lain
OUTPUT_DIR = Path("v2")                   # versi output run ini
SEED = 42
```

Path ditulis relatif terhadap folder notebook. Di notebook final, sel ini disederhanakan menjadi path relatif saja, sesuai aturan pengumpulan.

---

## 8. Pengumpulan

- Seno menyusun `final/`: `01_model.ipynb`, `02_submission.ipynb`, `model/`, `README.md`, `requirements.txt`.
- Isi `final/` + bobot model diunggah Seno ke **Drive**.
- Akses link: **siapa pun yang punya link bisa melihat**. Cek di mode incognito sebelum dikumpulkan.
