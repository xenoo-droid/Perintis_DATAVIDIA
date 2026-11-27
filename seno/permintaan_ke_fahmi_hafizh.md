# Permintaan dari C (Seno) ke B (Fahmi) & A (Hafizh) — Task 2

Jumat, 27 Nov 2026 · Deadline pengumpulan: **Sabtu, 28 Nov 23:59**

## Status C sekarang

- `seno/C_model.ipynb` + `seno/run1/` sudah jadi. Input: `hafizh/v2` + `fahmi/v1`. Prediksi keluar 6 kelas, siap jadi submit 1.
- `final/01_model.ipynb` & `final/02_submission.ipynb` sudah jadi kerangka. Bagian B (salinan kode Fahmi v1) sudah dicek: hasilnya sama 100% dengan `fahmi/v1`. Bagian A masih kosong, nunggu fungsi terkunci Hafizh.

## Temuan penting: `is_core` v1 cuma ada di 2 kelas

16 dari 20 cluster di `fahmi/v1` lemah, jadi `is_core` cuma ada di Explosion (20 foto) dan Tsunami (327 foto). Kalau label spreading hanya dari `is_core` (sesuai dokumen), classifier cuma bisa menebak 2 kelas: Tsunami 2.734, Explosion 446.

Usulan C: **`pusat_semua_cluster`**. Seed = 50% foto terdekat pusat dari **semua** cluster, aturannya sama dengan `is_core` Fahmi tapi tanpa filter cluster lemah. `rules.json` tidak diubah sama sekali. Hasil run1:

| Earthquake | Explosion | Fire | Hurricane | Tsunami | Volcano |
|---|---|---|---|---|---|
| 144 | 905 | 427 | 98 | 1.419 | 187 |

90% prediksi sama dengan nama cluster dari B.

## Kenapa fitur final penting

`final/01_model.ipynb` menjalankan ulang semuanya dari citra: fungsi A → kode B → kode C. Kalau fitur dari fungsi terkunci A beda dengan `hafizh/v2` (misalnya karena FP32 vs FP16, atau DINOv3), cluster B bisa berubah, dan submission yang sudah dikirim tidak bisa direproduksi oleh notebook final. Jadi submission final harus berasal dari fitur hasil fungsi terkunci. Itu sebabnya fungsi A dibutuhkan secepatnya.

---

## Untuk Fahmi (B)

| No | Permintaan | Kenapa | Target |
|---|---|---|---|
| 1 | Setuju atau tidak dengan `pusat_semua_cluster` untuk seed C. Kalau tidak, usulkan cara lain yang tetap tidak mengubah `rules.json`. | `is_core` v1 cuma ada di 2 kelas → classifier cuma bisa nebak 2 kelas | **Jumat malam**, sebelum submit 1 |
| 2 | Konfirmasi: `v1` final, atau bakal ada `v2`? Kalau ada v2, format output tetap sama (`events.parquet`, `event_names.csv`, `meta.json`) dan kabari Seno. | C dan `final/01` harus pakai versi B yang sama | Jumat malam |
| 3 | Siap jalanin ulang B di atas fitur baru dari fungsi terkunci Hafizh (folder versi baru), lalu push. | Fitur final bisa beda dari `hafizh/v2` → cluster bisa berubah | **Sabtu**, segera setelah fitur Hafizh ada |
| 4 | Kalau kode B berubah, kabari Seno. Bagian 2 `final/01_model.ipynb` adalah salinan B v1. | Notebook final harus sama persis dengan kode B yang dipakai | Setiap ada perubahan |
| 5 | Bahan write-up: alasan setiap aturan di `rules.json`, aturan NaN keluarga 1, dan aturan kelas kosong. | Tugas B no. 9, untuk README final | Sabtu siang |

Info aja (keputusan tetap di B, dan `rules.json` tidak diubah setelah lihat skor): kesepakatan dua keluarga aturan cuma 3 dari 14 cluster (21%), dan cluster 8 margin-nya 0,0007.

---

## Untuk Hafizh (A)

| No | Permintaan | Kenapa | Target |
|---|---|---|---|
| 1 | Pilih backbone final: **DINOv2-L** (`v2`) atau **DINOv3 SAT** (`v3.1`). | B v1 dan C run1 dibangun di atas `v2`. Kalau pindah ke DINOv3, B dan C harus diulang. | **Jumat malam** |
| 2 | Fungsi terkunci `ekstrak_semua(data_dir, ids)` sesuai kontrak di bawah, **FP32 + deterministik**, dalam satu sel yang bisa disalin persis. | Ditempel di 2 notebook final. `02_submission` wajib sama 100% dengan `01_model`; `v2` dan `v3.1` masih `fp16_autocast`. | **Secepatnya**, paling lambat Sabtu siang |
| 3 | Jalankan fungsi itu sekali, lalu push hasilnya ke folder versi baru: `dino_*.npy`, `artifact_and_phys_v3.parquet`, `patch_change_stats_v3.parquet`, `phys_v4.parquet`, `meta.json`. | B dan C diulang dengan fitur yang persis sama dengan notebook final, supaya submission final bisa direproduksi | Bareng no. 2 |
| 4 | Push `dino_v4_frozen_for_seno.py` dan `A_fitur.ipynb` yang disebut di `hafizh/v3.1/meta.json` (belum ada di repo). Jelaskan juga catatan "Run v3.2" padahal foldernya `v3.1`. | Supaya jelas kode mana yang menghasilkan versi mana | Jumat malam |
| 5 | Bobot offline: link Drive + path yang diharapkan fungsi (misalnya `weights/...`). Pastikan fungsi jalan **tanpa internet dan tanpa HF token**. | `extract_v4_meta.json` masih `loaded_from: online` | Sabtu siang |
| 6 | Daftar library + versi buat ekstraksi (torch, transformers/timm, opencv, scikit-image), plus perkiraan waktu ekstraksi 3.180 pasang di Kaggle T4. | Buat `requirements.txt` dan jadwal run final (01 dan 02 masing-masing ekstrak ulang dari citra) | Sabtu siang |

### Kontrak fungsi `ekstrak_semua`

```python
def ekstrak_semua(data_dir: Path, ids: np.ndarray) -> dict:
    # data_dir berisi <Id>/pre.jpg dan <Id>/post.jpg
    # ids = sample_submission["Id"]; urutan output WAJIB mengikuti ids
    return {
        "emb":   {"color_pre": ..., "color_post": ...},  # (N, 1024) float32; cabang gray boleh ditambah
        "art":   ...,  # DataFrame artifact_and_phys_v3, ada kolom Id
        "patch": ...,  # DataFrame patch_change_stats_v3, ada kolom Id
        "phys":  ...,  # DataFrame phys_v4, ada kolom Id
    }
```

Kolom yang dipakai B (namanya jangan diubah):

| Tabel | Kolom |
|---|---|
| `phys` | `d_dark`, `d_entropy`, `d_exg_v4`, `water_rel_pre`, `d_water`, `d_bright_raw_v4` |
| `art` | `delta_edge`, `edge_pre` |
| `patch` | `top10_share`, `radial_decay`, `com_shift`, `frac_chg_p15` |

---

## Urutan serah-terima

| Waktu | A — Hafizh | B — Fahmi | C — Seno |
|---|---|---|---|
| Jumat malam | No. 1, 4 | No. 1, 2 | Submit 1 (`run1`) setelah Fahmi oke |
| Sabtu pagi–siang | No. 2, 3, 5, 6 | No. 3 (ulang B di fitur baru), 5 | Ulang C → `run2` → submit 2; tempel fungsi A ke notebook final |
| Sabtu sore–malam | Standby kalau ada error | Standby | Run final 01 → 02 di Kaggle, cek 100%, upload ke Drive, kumpulkan link sebelum 23:59 |
