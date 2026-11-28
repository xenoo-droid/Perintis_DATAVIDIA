# Untuk Seno — run8_v4 (cluster v4 Fahmi)

## Cara pakai
1. `git pull`, lalu salin `fahmi/run_v4/C_model_run8_v4.ipynb` ke `seno/C_model_run8_v4.ipynb` (jangan timpa run7).
2. Jalankan semua sel. Output: `seno/run8_v4/submission_run8_v4.csv`.
3. Commit notebook + `run8_v4/`: `[C] run8_v4: LogReg, cluster v4 (changed), nama v4`.

## Yang berubah dari C_model_run7.ipynb (hanya 4 sel)
| Sel | Perubahan |
|---|---|
| 4 | `LABEL_DIR` dan `CLUSTER_DIR` = `../fahmi/v4_perubahan`; `RUN_PEMBANDING = "run7_revisi2"`; `OUTPUT_DIR = "run8_v4"`; `FITUR_BLOK` + `"changed"`; baru: `B_BLOK = ["changed"]`, `FILE_EMB = {"changed": "dino_changed_embed"}` |
| 6 | Embedding dimuat untuk `FITUR_BLOK` + `B_BLOK`, nama file lewat `FILE_EMB` |
| 10 | Ruang Z: `X_b = np.hstack([l2(emb[b]) for b in B_BLOK])` (v3: color_pre + color_post) |
| 20 | Teks `strategi` di meta.json menyebut v4 |

Sisanya sama dengan run7 (seed kedekatan 0,5; yakin 1,0 / ragu 0,5; tanpa nama 0; LS kNN 15 alpha 0,2; PCA 128; LR C 1,0 balanced).

## Yang harus terlihat
- Sel 6: "Cluster tanpa nama ... [1, 2, 4, 5, 13, 17, 21, 22, 25, 29, 30, 32]" (12 cluster, bobot 0). Ini disengaja: cluster yang ciri kerusakannya hanya dari warna.
- Sel 10: "Centroid B tereproduksi: selisih maks ~1.5e-03" (lolos, batas 5e-3).
- Sel 14: CV macro-F1 ±0,845. Sel 15: 6 kelas terprediksi, sama dengan run7 ±59%.
- Distribusi: Earthquake 1101, Explosion 104, Fire 327, Hurricane 1401, Tsunami 90, Volcano 157.

Diuji di laptop Fahmi (scikit-learn 1.7.2): hasil identik dengan `fahmi/run_v4/submission_run_v4.csv`.
Jika versi scikit-learn beda, angka bisa sedikit bergeser; yang penting sel 10 lolos.

## Catatan untuk final/
`final/01_model.ipynb` masih v3. Jika run8_v4 dipilih: bagian B memuat `fahmi/v4_perubahan` (hash `event_id_sha1` di meta.json v4,
tanpa sub_events), ruang Z dari `changed`, `FITUR_BLOK` + `changed`, dan fungsi A dipanggil dengan `compute_patch_and_phys=True`
supaya `emb["changed"]` terbentuk.
