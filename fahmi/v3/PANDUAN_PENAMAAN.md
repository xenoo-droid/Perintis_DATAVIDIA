# Panduan penamaan cluster — STRATEGI v3

Dasar: FAQ panitia no. 7 — cluster dibuat model, **nama diberikan manual oleh tim per cluster** dari pengamatan citra,
pemetaan ditulis eksplisit di notebook, alasannya dijelaskan di write-up. Label per foto dilarang (no. 6).
AI tidak boleh dipakai untuk menamai cluster (no. 4).

## File
- **`lembar_bersama.csv` — satu-satunya lembar kerja.** Dinilai bersama oleh tim. Kolom: `event_id, n, montase, kelas_final, yakin, alasan`.
  Jangan menambah/menghapus baris atau mengubah `event_id`, `n`, `montase`. Baris sub-cluster baru (mis. `7a`, `7b`) ditambahkan oleh kode setelah pemecahan cluster campur.
- `montase/cluster_XX.jpg` — warna asli; `montase/cluster_XX_norm.jpg` — warna dinormalisasi (gray-world + stretch).
- Tiap montase: 16 pasangan, **kiri = pre, kanan = post**. 4 baris atas = pasangan terdekat pusat cluster,
  4 baris bawah = pasangan acak dari cluster yang sama (untuk melihat keragaman cluster).
- Id di pojok tile hanya untuk referensi diskusi, **bukan** untuk melabeli foto.
- **Arsip:** lembar per orang, `lembar_penamaan.csv`, `lembar_putaran1/`, dan `cadangan_sebelum_putaran2/` sudah dihapus dari folder ini. Isinya masih ada di riwayat git (putaran 1: commit `c255fc5` dan `950440b`; putaran 2 per orang: `2ed1593`/`c949a04`). Kappa putaran 1 dan 2 ada di `meta.json`.

## Cara bekerja (satu lembar, bersama)
1. Buka `lembar_bersama.csv` dan montase di layar yang sama. **Satu orang saja yang mengetik**, yang lain menyimak montase.
2. Baris yang sudah terisi (hasil putaran 1–2 yang disepakati/yakin) boleh langsung dipakai. Kalau ada yang ingin diubah, ubah di sini setelah diskusi.
3. Baris yang `kelas_final`-nya kosong: buka **kedua** versi montase, diskusikan, lalu isi.
4. Kalau tetap tidak sepakat, **orang ketiga** (yang belum menilai cluster itu di putaran sebelumnya) menjadi penentu.
5. Isian yang sah:
   - `kelas_final`: `Earthquake`, `Explosion`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, atau `campur`.
   - `yakin`: `yakin` atau `ragu`. Wajib diisi jika `kelas_final` diisi. Dipakai Seno sebagai bobot latih (yakin = 1,0; ragu = 0,5).
   - `alasan`: **1 kalimat** tentang ciri visual yang terlihat di montase. Wajib (FAQ no. 7), dipakai di write-up.
6. `campur` dipakai jika pasangan di montase jelas terbagi ke dua jenis kejadian atau lebih. Fahmi menentukan jumlah bagian (default 2);
   cluster dipecah model, lalu sub-cluster-nya dinamai di lembar yang sama.
7. Simpan sebagai **CSV UTF-8 dengan pemisah koma** (Excel: *Save As → CSV UTF-8 (Comma delimited)*). Jangan pakai titik koma.
8. Sumber tunggal adalah `lembar_bersama.csv`. Jangan membuat ulang lembar per orang atau menjalankan kode penggabung lamanya.

### Contoh `alasan` (format 1 kalimat, ciri visual saja)
- "Pada sebagian besar pasangan, post menunjukkan area gelap dan gersang menggantikan vegetasi yang hijau di pre."
- "Hampir semua pasangan berada di permukiman pesisir; di post daratan tergenang dan garis pantai berubah."
- "Setengah pasangan menunjukkan kerusakan bangunan terpusat, setengah lainnya lahan tertutup material abu-abu (campur)."

Sebut ciri yang terlihat pada **mayoritas** pasangan, bukan satu foto. **Jangan menyebut nama kejadian atau lokasi tertentu**
(nama gempa, badai, kebakaran, gunung): itu bisa terkesan berasal dari dataset berlabel (xBD/BRIGHT), bukan dari apa yang terlihat.

## Dua jalur setelah lembar diisi

Isi `lembar_bersama.csv` sampai **penuh** dulu (tidak ada `kelas_final` kosong). Setelah itu ada dua jalur, tergantung ada tidaknya baris `campur`.

**Jalur 1 — tidak ada cluster campur**
1. Lembar penuh, tidak ada baris `campur`.
2. Fahmi menjalankan **Prompt B** (v3-7, finalisasi) di Claude Code.
3. Selesai: `cluster_names.json` dan `events.parquet` siap untuk Seno.

**Jalur 2 — ada cluster campur**
1. Lembar penuh. Baris `campur` boleh ada, dan daftarnya harus sama persis dengan `CLUSTER_CAMPUR` yang ditentukan Fahmi (`{id: jumlah_bagian}`).
2. Fahmi menjalankan **Prompt A** (v3-6, pecah cluster campur). Claude Code lalu **berhenti dan melapor**.
3. Tim menamai sub-cluster di lembar yang sama (lihat bagian berikut).
4. Fahmi menjalankan **Prompt B** (v3-7, finalisasi).
5. Selesai.

Kalau ada yang diubah setelah selesai: label cluster biasa atau sub-cluster cukup diedit di lembar lalu Prompt B diulang. Kalau yang berubah status campur atau jumlah bagiannya, ulang dari Prompt A. Simpan salinan lembar sebelum mengubah, dan ubah hanya berdasarkan pengamatan montase.

Cadangan kalau waktu mepet: beri tiap cluster campur **satu kelas dominan** langsung di lembar (tanpa dipecah), lalu jalankan Prompt B. Kualitas label cluster itu lebih rendah, jadi tandai `yakin` = `ragu`.

## Cluster campur dan sub-cluster

**Berapa bagian.** Bebas, tidak harus 2. Fahmi menentukan per cluster lewat `CLUSTER_CAMPUR`, mis. `{2: 2, 10: 3, 29: 2, 39: 3}`. Default 2; cluster besar lebih masuk akal dibagi 3.

**Siapa yang membagi foto.** **Model**, bukan manusia. Kode mengambil semua foto di cluster itu, menghitung kemiripan antar foto dari embedding yang sama, lalu Spectral Clustering (seed 42) membaginya menjadi k kelompok. Tiap foto dapat ID sub-cluster (`2a`, `2b`, …). Tim **tidak** memilih foto satu per satu (itu label per foto, dilarang FAQ no. 6); tim hanya **menamai** kelompok yang sudah terbentuk.

**Apa yang berubah di lembar.** Tidak ada file baru. Kode menambah baris `2a`, `2b`, … ke `lembar_bersama.csv` (kolom `kelas_final`, `yakin`, `alasan` kosong). Baris induk (`2`) tetap ada dengan `kelas_final` = `campur` dan tidak dipakai sebagai label latih. Montase sub-cluster (`cluster_2a.jpg` dan `_norm`) dibuat di `montase/`. Sub-cluster saudara boleh berkelas sama.

**Cara menamai sub-cluster.** Sama seperti cluster biasa: nilai **mayoritas** pasangan di montase, isi `kelas_final`, `yakin`, `alasan` (ciri visual saja).

**Kalau ada kelas yang tidak terlihat.** Montase hanya menampilkan 16 foto, sedangkan cluster berisi 46–137 foto, jadi kelas ketiga bisa tersembunyi.
- Montase sub-cluster berisi 16 foto **baru**, sehingga kelas tersembunyi sering muncul di sana.
- Kalau sub-cluster masih terlihat campur, pilih salah satu:
  - ulangi Prompt A dengan jumlah bagian lebih banyak (mis. 2 jadi 3), atau
  - beri kelas yang paling dominan dan tandai `yakin` = `ragu` (bobot latih jadi separuh).
- Sisa derau label tidak bisa dihilangkan sepenuhnya. Classifier cukup toleran, dan keterbatasan ini ditulis jujur di write-up.

**Catatan penting.** Model membagi foto menurut **kemiripan tampilan** (lanskap, sensor, tekstur), **bukan** menurut jenis bencana. Cluster campur bisa terbelah menurut lanskap dan belum tentu sejalan dengan kelas. Karena itu sub-cluster tetap harus dilihat langsung di montase, jangan diasumsikan sudah bersih.

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
- Perubahan nama setelah diisi hanya lewat diskusi tim berdasarkan pengamatan montase, dan alasannya diperbarui di kolom `alasan`.
