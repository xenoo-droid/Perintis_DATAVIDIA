# Untuk Seno — serah terima Fahmi (v3-7, final)

Penamaan cluster sudah final. Semua file ada di `fahmi/v3/`. Tidak ada yang perlu kamu hitung ulang dari sisi clustering.

## 1. Isi yang kamu pakai
| File | Fungsi |
|---|---|
| `events.parquet` | 3180 baris urutan sample_submission. Kolom: Id, event_id (0-49, 46 ID dipakai), jarak_pusat, kedekatan (0-1), kelas_cluster, yakin_cluster |
| `cluster_names.json` | kunci "0".."49" (tanpa 2, 10, 29, 39) -> {kelas, yakin, alasan, penilai, sub_id} |
| `dictionary_nama_cluster.py` | kamus eksplisit siap salin ke notebook final (FAQ no. 7) |
| `sub_events.parquet` | pemetaan foto -> induk/sub (hanya untuk cek hash) |
| `meta.json` | ringkasan (bagian `finalisasi`) |

Ringkasan: 46 cluster. Earthquake 8 (572 foto), Fire 12 (892), Hurricane 18 (1395), Tsunami 4 (155), Volcano 4 (166), Explosion 0. 19 yakin, 27 ragu. Penilai semua "AI (Claude)", tanpa tinjauan tim.

## 2. Perubahan di `seno/C_model.ipynb` (run5)
Duplikat dulu notebook jadi `C_model_run5.ipynb` (jangan timpa yang lama). Di sel pengaturan (sel 4):
```
LABEL_DIR = "../fahmi/v3"
CLUSTER_DIR = "../fahmi/v3"
RUN_PEMBANDING = "run4_sementara2"
OUTPUT_DIR = "run5_final"
```
Sisanya tetap (SEED_KEDEKATAN_MIN=0.5, BOBOT_YAKIN yakin 1.0 / ragu 0.5, LS kNN 15 alpha 0.2, LR C_PCA 128 LR_C 1.0 balanced).

Yang harus terlihat saat dijalankan:
- sel 6: assert `set(events.event_id) == set(names.index)` lolos, 46 ID, tidak ada cluster tanpa nama.
- sel 10: peringatan bentuk centroid (40,64) vs (46,64) muncul. Ini wajar karena centroids.npy tetap 40 cluster. Hanya peringatan, jangan dihentikan.
- event_id TIDAK dipakai sebagai fitur (tetap seperti sebelumnya).

## 3. Perubahan di `final/01_model.ipynb`
1. Sel 4: `MODE_UJI = False`, hapus `RULES_PATH`/`B_K`/path acuan lama.
2. Ganti bagian 2 (clustering+penamaan berbasis rules.json) dengan:
   - salin `events.parquet`, `sub_events.parquet`, `cluster_names.json` ke `final/` (pakai path relatif di sel pengaturan),
   - tempel isi `dictionary_nama_cluster.py` sebagai sel kamus eksplisit,
   - `kelas_cluster = event_id.map(NAMA_CLUSTER)`, assert sama dengan kolom `kelas_cluster` di parquet.
3. Cek integritas (jangan `.fit()` ulang clustering di Kaggle): rekonstruksi event_id induk 40-cluster lalu hash:
```
import hashlib, pandas as pd
ev = pd.read_parquet("events.parquet").set_index("Id")
sub = pd.read_parquet("sub_events.parquet")
induk = ev["event_id"].copy()
induk.loc[sub["Id"].values] = sub["event_id_induk"].values
assert hashlib.sha1(induk.values.astype("int32").tobytes()).hexdigest() == EVENT_ID_SHA1_40CLUSTER
```
   (sudah saya uji: cocok 9517d553...; `int64` TIDAK cocok, wajib int32).
4. Bagian seed/LS/LR (sel 12-13) tetap, pakai parameter yang sama dengan run5.
5. Simpan model ke `final/model/`.

## 4. Perubahan di `final/02_submission.ipynb`
- Tidak ada `.fit()`, tidak ada pencocokan centroid ke cluster (nearest-centroid hanya ~84,5% cocok, jangan dipakai).
- Muat model tersimpan, tempel fungsi terkunci Hafizh, prediksi, tulis submission. Hasil harus identik dengan file yang kamu pilih.

## 5. README & requirements
- `final/README.md`: hapus klaim "tanpa label manusia" / "rules.json". Tulis: nama kelas per cluster ditentukan lewat pengamatan montase oleh AI (Claude), kamus eksplisit ada di notebook 01, tanpa tinjauan tim, tidak disetel dari leaderboard.
- `final/requirements.txt`: pin versi (numpy, pandas, pyarrow, scikit-learn, joblib, torch/transformers sesuai DINOv2-L).
- Bobot DINOv2 offline di Drive, link terbuka.

## 6. Catatan
- Jangan ubah nama kelas berdasarkan skor leaderboard (aturan FAQ 7).
- Hanya 1 submission final, maks 3/hari. Catat di `seno/log_submission.csv`.
- Risiko diketahui: tidak ada cluster Explosion (Macro F1 maksimal sekitar 0,83), 27 dari 46 cluster berstatus ragu.
