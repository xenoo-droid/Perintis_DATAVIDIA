# Panduan penamaan cluster — STRATEGI v3

Dasar: FAQ panitia no. 7 — cluster dibuat model, **nama diberikan manual oleh tim per cluster** dari pengamatan citra,
pemetaan ditulis eksplisit di notebook, alasannya dijelaskan di write-up. Label per foto dilarang (no. 6).
AI tidak boleh dipakai untuk menamai cluster (no. 4).

## File
- `lembar_<nama>.csv` — **lembar milikmu**, hanya berisi cluster tugasmu. Isi kolom `kelas`, `yakin`, `alasan` (dan `catatan` jika perlu).
  Jangan menambah/menghapus baris atau mengubah kolom `event_id` dan `peran`.
- `montase/cluster_XX.jpg` — warna asli; `montase/cluster_XX_norm.jpg` — warna dinormalisasi (gray-world + stretch).
- Tiap montase: 16 pasangan, **kiri = pre, kanan = post**. 4 baris atas = pasangan terdekat pusat cluster,
  4 baris bawah = pasangan acak dari cluster yang sama (untuk melihat keragaman cluster).
- Id di pojok tile hanya untuk referensi diskusi, **bukan** untuk melabeli foto.
- `lembar_penamaan.csv` — gabungan otomatis dari ketiga lembar (dibuat kode). Penilai tidak perlu membukanya.

## Pembagian tugas
| Penilai | Lembar | Penilai 1 (cluster) | Penilai 2 / cek silang (cluster) | Total |
|---|---|---|---|---|
| Fahmi | `lembar_Fahmi.csv` | 03, 08, 11, 14, 16, 17, 18, 26, 27, 28, 32, 33, 35, 36 | 06, 12, 20 | 17 |
| Hafizh | `lembar_Hafizh.csv` | 00, 01, 04, 05, 06, 07, 21, 22, 23, 24, 31, 34, 38 | 15, 17, 28 | 16 |
| Seno | `lembar_Seno.csv` | 02, 09, 10, 12, 13, 15, 19, 20, 25, 29, 30, 37, 39 | 01, 05, 23, 27 | 17 |

Cek silang: cluster yang dinilai dua orang secara **terpisah**. Lembarmu tidak memuat jawaban orang lain — jangan saling
memberi tahu jawaban untuk cluster cek silang sebelum keduanya selesai.

## Cara mengisi
1. Untuk setiap baris di lembarmu, buka **kedua** versi montase. Nilai seluruh pasangan, bukan satu foto.
2. Isian yang sah:
   - `kelas`: `Earthquake`, `Explosion`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, atau `campur` (huruf besar/kecil tidak masalah).
   - `yakin`: `yakin` atau `ragu`. Wajib diisi jika `kelas` diisi.
   - `alasan`: **1 kalimat** tentang ciri yang terlihat di montase.
   - `catatan`: opsional.
3. `campur` dipakai jika pasangan di montase jelas terbagi ke dua jenis kejadian atau lebih.
   Fahmi menentukan jumlah bagian (default 2); cluster dipecah model, lalu sub-cluster-nya dinamai lagi.
4. Simpan tetap sebagai **CSV**. Dari Excel: *Save As → CSV UTF-8 (Comma delimited)*. Pemisah `;` juga bisa dibaca.
5. Kirim/push lembarmu ke `fahmi/v3/` dengan nama file yang sama.

### Contoh `alasan` (format 1 kalimat)
- "Pada sebagian besar pasangan, post menunjukkan area gelap dan gersang menggantikan vegetasi yang hijau di pre."
- "Hampir semua pasangan berada di permukiman pesisir; di post daratan tergenang dan garis pantai berubah."
- "Setengah pasangan menunjukkan kerusakan bangunan terpusat, setengah lainnya lahan tertutup material abu-abu (campur)."

Sebut ciri yang terlihat pada **mayoritas** pasangan, bukan satu foto. Alasan ini dipakai langsung di write-up.

## Bagaimana `kelas_final` ditentukan
- kelas_final diisi otomatis oleh kode jika (a) kelas penilai 1 = kelas penilai 2, atau (b) cluster tanpa cek silang dan yakin_1 = 'yakin'. Baris lain (beda pendapat atau ragu) dikosongkan dengan catatan 'perlu diskusi'; Fahmi mengisi kelas_final setelah diskusi tim.
- Isian manual Fahmi di `lembar_penamaan.csv` dipertahankan saat penggabung dijalankan ulang.
- Jumlah bagian cluster campur ditentukan Fahmi (default 2); cluster dipecah model lalu sub-cluster dinamai tim. Jika waktu habis, cluster campur diberi satu kelas dominan menurut penilai.

## Ciri kelas (deskripsi resmi panitia)
| Kelas | Deskripsi |
|---|---|
| Earthquake | Guncangan di kerak bumi; pola kerusakan membentang luas |
| Fire | Kebakaran vegetasi/lahan/bangunan; area terbakar gelap dan gersang |
| Hurricane | Badai tropis berangin kencang; merontokkan vegetasi, merusak struktur secara tersebar |
| Tsunami | Gelombang laut menggenangi daratan; mengubah garis pantai, menyeret material |
| Volcano | Letusan gunung api; material vulkanik menutupi permukiman dan lahan |
| Explosion | Ledakan besar; merusak area sekitar titik kejadian secara radial |

## Gangguan yang BUKAN ciri bencana (augmentasi dataset)
- **Warna diacak**: tint pink, biru, hijau, atau gelap. Bandingkan dengan versi `_norm`; jangan menyimpulkan air/api/abu dari warna saja.
- **Grayscale**: sebagian gambar tidak berwarna.
- **Rotasi, flip, warp**: pre dan post tidak sejajar; bandingkan isi, bukan posisi piksel.
- **Area hitam**: bagian kosong di tepi/potongan gambar.
- **Scanline**: garis-garis horizontal/vertikal. Cluster bisa terlihat seragam karena scanline — abaikan garisnya, nilai isi citranya.
- **Blur, awan, kabut**.

## Aturan
- **Satu keputusan per cluster.** Tidak ada label per foto, termasuk "foto ini beda, kelasnya X".
- Nilai hanya dari citra di montase. Jangan mencari Id/citra di internet (reverse image search, lokasi kejadian, dsb.).
- **Jangan** meminta AI (ChatGPT, Claude, Gemini, CLIP, dsb.) menamai cluster atau menilai montase, dan jangan mengunggah montase ke AI.
- **Jangan** mengubah nama cluster karena skor leaderboard atau karena distribusi kelas yang dihasilkan.
- Setelah lembar diserahkan, perubahan nama hanya lewat diskusi tim dan dicatat di `catatan`.
