# AdaptiveLab - Studio Segmentasi Citra Dokumen Berbasis Adaptive Gaussian Thresholding

AdaptiveLab adalah aplikasi web interaktif yang dikembangkan dari awal (*from scratch*) untuk memecahkan permasalahan binerisasi dan segmentasi citra dokumen dengan degradasi pencahayaan tidak merata, bayangan kuat (*shadows*), tinta tembus (*bleed-through*), kontras rendah, dan gangguan *noise*. Proyek ini dirancang sebagai platform riset dan tugas kuliah *Computer Vision* yang dioptimalkan untuk dataset benchmark naskah kuno internasional **H-DIBCO 2016** (https://vc.ee.duth.gr/h-dibco2016/), dilengkapi eksperimen *grid search*, perbandingan multi-metode, serta evaluasi kuantitatif terhadap *Ground Truth*.

---

## 1. Fitur Utama

1. **Inti Metode Tunggal: Adaptive Gaussian Thresholding**
   - Mendukung dua mesin eksekusi:
     - **OpenCV Engine:** `cv2.adaptiveThreshold` dengan bendera `ADAPTIVE_THRESH_GAUSSIAN_C` dan `THRESH_BINARY`.
     - **Manual Scratch Engine:** Implementasi manual dari rumus matematika dengan konvolusi kernel sebaran Gaussian 2D kustom yang dibangun dari fungsi sebaran normal ($\sigma = 0.3 \times ((N - 1) \times 0.5 - 1) + 0.8$) dan dinormalisasi ($\sum K = 1.0$), menghasilkan selisih piksel $< 0.3\%$ (hampir identik) terhadap OpenCV.
   - Peta Ambang Lokal $T(x, y)$ divisualisasikan secara langsung.

2. **Fleksibilitas Pipeline Pengolahan Citra**
   - **Pra-pemrosesan (Opsional):** Konversi Grayscale, Gaussian Blur (penghalusan variasi mikro), Median Blur (penghapusan noise impulsif *salt-and-pepper*), dan CLAHE (*Contrast Limited Adaptive Histogram Equalization* untuk perataan kontras lokal).
   - **Pasca-pemrosesan (Opsional):** Morfologi Matematika (*Opening*, *Closing*, *Erosion*, *Dilation* dengan ukuran kernel, iterasi, dan bentuk elemen penstruktur yang dapat diatur), serta *Remove Small Connected Components* (penghapusan bintik hitam/putih berukuran $< \text{min\_area}$).
   - **Indikator Status Pipeline:** Visualisasi alur tahap demi tahap secara dinamis.

3. **Komparasi 4 Metode Berdampingan**
   - Membandingkan hasil pada citra yang sama antara:
     1. Global Thresholding (Ambang konstan $T$ manual)
     2. Otsu's Thresholding (Ambang global optimal bimodal otomatis)
     3. Adaptive Mean Thresholding (Rata-rata aritmatika lokal)
     4. Adaptive Gaussian Thresholding (Rata-rata berbobot Gaussian lokal)

4. **Mode Eksperimen Grid Search & Ekspor Riset**
   - Uji kombinasi parameter daftar Block Size ($N$) dan daftar konstanta ($C$).
   - Menyajikan tabel rekapitulasi metrik, persentase piksel, durasi komputasi (ms), dan galeri *thumbnail*.
   - Ekspor tabel ke file **CSV** dan seluruh citra hasil resolusi penuh ke dalam arsip **ZIP**.

5. **Evaluasi Kuantitatif Terhadap Ground Truth**
   - Pengujian berbasis piksel teks (Foreground) vs latar (Background).
   - Metrik lengkap: **Precision**, **Recall**, **F1-Score**, **Accuracy**, **Peak Signal-to-Noise Ratio (PSNR)**, dan **Mean Squared Error (MSE)**.
   - **Confusion Color Map (Peta Visual Selisih):**
     - 🟩 **True Positive (TP):** Teks terdeteksi secara tepat.
     - 🟥 **False Positive (FP):** Latar/noise yang keliru dideteksi sebagai teks.
     - 🟦 **False Negative (FN):** Bagian teks asli yang terlewat / terputus.
     - ⬜ **True Negative (TN):** Latar belakang bersih.
   - Penyelarasan dimensi otomatis (*nearest-neighbor*) disertai peringatan jika ukuran *Ground Truth* berbeda.

6. **Interaktivitas & Antarmuka Modern**
   - *Split-screen comparison slider* (sebelum & sesudah binerisasi) yang interaktif.
   - Grafik histogram 256 tingkat keabuan berbasis Chart.js.
   - Dukungan tema otomatis (Sistem, Terang, Gelap) dengan palet profesional non-generic.
   - Pembaruan otomatis (*debounced realtime updates*) tanpa memuat ulang (*reload*) halaman.

---

## 2. Struktur Proyek

```text
AdaptiveLab/
├── app.py                      # Aplikasi Flask & routing API JSON
├── requirements.txt            # Dependensi paket Python
├── README.md                   # Dokumentasi teknis & panduan riset
├── core/                       # Modul inti pengolahan citra & Computer Vision
│   ├── __init__.py
│   ├── threshold.py            # Adaptive Gaussian (OpenCV, Manual Scratch, Map, Validasi)
│   ├── preprocess.py           # Grayscale, Gaussian Blur, Median Blur, CLAHE
│   ├── postprocess.py          # Morfologi Matematika & Penyaringan Komponen Kecil
│   ├── compare.py              # Pembanding 4 metode (Global, Otsu, Mean, Gaussian)
│   ├── metrics.py              # Evaluasi kuantitatif (Precision, Recall, F1, PSNR, Diff Map)
│   ├── experiment.py           # Parameter Grid Search, Generator CSV & ZIP
│   └── utils.py                # Decode bytes aman, Base64 encoder, Histogram, CC Stats
├── templates/
│   └── index.html              # Antarmuka SPA (HTML5 semantik & aksesibel)
├── static/
│   ├── css/
│   │   └── style.css           # Desain modern, palet slate-teal, tema terang/gelap
│   └── js/
│       ├── api.js              # Klien HTTP Fetch ke backend Flask
│       ├── viewer.js           # Slider interaktif Before/After
│       ├── charts.js           # Visualisasi kurva histogram Chart.js
│       └── main.js             # State management, debounce, navigasi tab, modal
├── sample_images/              # Citra uji sintetis & Ground Truth biner
│   ├── sample1_uneven_light.png
│   ├── sample1_uneven_light_gt.png
│   ├── sample2_salt_pepper.png
│   ├── sample2_salt_pepper_gt.png
│   ├── sample3_low_contrast.png
│   └── sample3_low_contrast_gt.png
├── scripts/
│   ├── make_samples.py         # Skrip pembuatan dataset sintetis terdegradasi
│   └── generate_report_assets.py # Skrip pembangkit aset gambar & tabel untuk Bab 4 laporan
├── report_assets/              # Aset siap pakai untuk Bab 4 laporan kuliah
│   ├── 01_original.png ... 08_confusion_diff_map.png
│   ├── compare_montage_2x2.png
│   ├── tabel_eksperimen_blocksize_vs_c.csv
│   ├── histogram_intensitas.png
│   └── grafik_f1_vs_blocksize.png
├── outputs/                    # Direktori penyimpanan sementara server
└── tests/                      # Rangkaian pengujian otomatis Pytest
    ├── test_threshold.py       # Validasi parameter & uji identik manual vs opencv (< 1%)
    ├── test_modules.py         # Pengujian preprocess, postprocess, metrics, compare
    └── test_api.py             # Pengujian endpoint REST API Flask
```

---

## 3. Cara Instalasi dan Menjalankan di Windows

Pastikan Python 3.10 atau versi yang lebih baru telah terpasang di sistem Windows Anda.

### Langkah 1: Buka Terminal (PowerShell atau Command Prompt)
Navigasikan ke direktori proyek:
```powershell
cd "d:\Antigravity Project\AdaptiveLab"
```

### Langkah 2: Buat & Aktifkan Lingkungan Virtual (Virtual Environment)
```powershell
py -m venv .venv
.venv\Scripts\activate
```

### Langkah 3: Pasang Dependensi
```powershell
pip install -r requirements.txt
```

### Langkah 4: Jalankan Server Aplikasi
```powershell
py app.py
```

### Langkah 5: Buka Aplikasi di Peramban Web
Buka URL berikut pada peramban (Chrome, Edge, Firefox):
```text
http://127.0.0.1:5050
```

---

## 4. Cara Menjalankan Pengujian Otomatis (Pytest)

Aplikasi dilengkapi unit test komprehensif yang memvalidasi kebenaran matematis, toleransi perbedaan algoritma manual vs OpenCV, serta endpoint API:

```powershell
python -m pytest -v
```

Hasil pengujian mencakup:
- Validasi tipe dan batasan parameter ($N \ge 3$ ganjil, $C \in \mathbb{Z}$).
- Pembuktian bahwa selisih piksel `adaptive_gaussian_manual` dan `adaptive_gaussian_opencv` $< 1.0\%$ (bahkan $\le 0.27\%$).
- Integritas perhitungan metrik Precision, Recall, F1-Score, dan PSNR.
- Respons status code dan payload endpoint REST API Flask.

---

## 5. Cara Membuat Sampel Sintetis & Aset Laporan

### Membuat Citra Sampel Pengujian Baru:
```powershell
python scripts/make_samples.py
```
Perintah ini akan membuat 3 pasang citra dokumen teks sintetis dan berkas acuan *Ground Truth* di folder `sample_images/`:
1. `sample1_uneven_light`: Gradien pencahayaan diagonal kuat dan bayangan lipatan buku.
2. `sample2_salt_pepper`: Gangguan *noise* impulsif *salt-and-pepper* (3.5%).
3. `sample3_low_contrast`: Dokumen pudar dengan selisih kontras teks dan latar hanya 25 level keabuan.

### Membangkitkan Aset untuk Bab 4 Laporan:
```powershell
python scripts/generate_report_assets.py
```
Perintah ini mengeksekusi pipeline lengkap dan menyimpan seluruh materi siap pakai ke direktori `report_assets/`:
- Gambar tahapan pipeline (Asli $\rightarrow$ Grayscale $\rightarrow$ Blur $\rightarrow$ Peta Ambang $\rightarrow$ Biner $\rightarrow$ Bersih $\rightarrow$ Confusion Map).
- Kolase perbandingan 4 metode (`compare_montage_2x2.png`).
- Tabel CSV rekapitulasi kombinasi parameter (`tabel_eksperimen_blocksize_vs_c.csv`).
- Grafik kurva distribusi histogram dan pengaruh Block Size vs F1-score (`histogram_intensitas.png`, `grafik_f1_vs_blocksize.png`).

---

## 6. Penggunaan Dataset H-DIBCO 2016 (Handwritten Document Image Binarization Contest)

Aplikasi ini dioptimalkan dan diuji menggunakan dataset standar kompetisi internasional **H-DIBCO 2016** (*Handwritten Document Image Binarization Contest*):
- **Situs Resmi:** [https://vc.ee.duth.gr/h-dibco2016/](https://vc.ee.duth.gr/h-dibco2016/)
- **Penyelenggara:** Democritus University of Thrace (DUTH) & ICFHR (International Conference on Frontiers in Handwriting Recognition) 2016.

### A. Karakteristik & Tantangan Dataset H-DIBCO 2016
Dataset H-DIBCO 2016 terdiri dari dokumen naskah tulisan tangan kuno bersejarah yang memiliki berbagai degradasi kompleks:
1. **Tinta Tembus (*Bleed-Through / Show-Through*):** Goresan tulisan tinta dari halaman belakang menembus ke halaman depan kertas lapuk.
2. **Noda Kelembapan & Jamur (*Foxing*):** Bercak-bercak cokelat/kekuningan akibat degradasi kimiawi kertas kuno.
3. **Pencahayaan & Bayangan Tidak Merata:** Variasi iluminasi akibat kelengkungan jilidan naskah atau kondisi pemindaian/fotografi arsip.
4. **Intensitas Tinta Pudar (*Faded Ink*):** Goresan pena tipis yang mengalami penurunan kontras terhadap latar kertas.

### B. Langkah Menambahkan Dataset H-DIBCO 2016 ke AdaptiveLab
Untuk memasukkan citra dokumen H-DIBCO 2016 dan Ground Truth-nya:
1. Unduh berkas citra uji (*test images*) dan *Ground Truth* dari [https://vc.ee.duth.gr/h-dibco2016/](https://vc.ee.duth.gr/h-dibco2016/).
2. Salin berkas citra dokumen dan Ground Truth ke dalam folder:
   ```text
   d:\Antigravity Project\AdaptiveLab\sample_images\
   ```
3. Gunakan konvensi penamaan akhiran `_gt` untuk berkas Ground Truth agar sistem **otomatis mendeteksi dan memasangkannya**:
   - Citra Asli: `hdibco2016_01.png` (atau format `.bmp`, `.jpg`)
   - Citra Ground Truth: `hdibco2016_01_gt.png` (atau format `.bmp`)
4. Jalankan atau muat ulang aplikasi (`py app.py`). Berkas akan langsung terdaftar di menu pilihan sampel di web UI beserta indikator status `Ground Truth: Tersedia ✓`.

### C. Rekomendasi Parameter Pengujian untuk Dokumen H-DIBCO 2016
Berdasarkan karakteristik naskah tulisan tangan H-DIBCO 2016:
- **Block Size ($N$):** Gunakan $N = 25$ hingga $35$ (ganjil). Rentang ini ideal untuk melingkupi lebar goresan pena tulisan tangan serta spasi antar-kata.
- **Konstanta $C$:** Disarankan menaikkan nilai $C$ ke rentang **$8$ hingga $15$** (default: 5). Nilai $C$ yang lebih tinggi efektif menekan noise *bleed-through* (tinta tembus) dan noda kertas agar tidak salah terklasifikasi sebagai teks.
- **Pra-pemrosesan:** Aktifkan **Gaussian Blur** ($3 \times 3$, $\sigma = 0.8$) atau **Median Blur** ($3 \times 3$) untuk menghaluskan tekstur serat kertas kuno sebelum binerisasi.
- **Pasca-pemrosesan:** Aktifkan **Hapus Komponen Kecil** (*min area* $8 - 15$ px) untuk membersihkan bintik-bintik tinta tembus terisolasi.

### D. Format Evaluasi Standar & Atribusi Ilmiah
AdaptiveLab menghitung metrik kuantitatif resmi yang selaras dengan standar kompetisi DIBCO:
- **F-Measure (F1-Score):** Rata-rata harmonik Precision dan Recall deteksi goresan teks.
- **PSNR (Peak Signal-to-Noise Ratio):** Tingkat kemiripan visual terhadap Ground Truth biner.
- **Confusion Map:** Memvisualisasikan piksel teks terdeteksi benar (hijau), noise/bleed-through bocor (merah), dan teks terputus (biru).

> **Sitasi Ilmiah untuk Laporan Tugas/Makalah:**  
> I. Pratikakis, K. Zagoris, G. Barlas, and B. Gatos, *"ICFHR2016 Handwritten Document Image Binarization Contest (H-DIBCO 2016)"*, in Proceedings of the 15th International Conference on Frontiers in Handwriting Recognition (ICFHR), pp. 619-623, 2016. URL: [https://vc.ee.duth.gr/h-dibco2016/](https://vc.ee.duth.gr/h-dibco2016/)

---

## 7. Catatan Implementasi & Analisis Keputusan Desain (Untuk Bahan Laporan)

Bagian ini dirancang khusus untuk mempermudah penulisan **Bab 3 (Metodologi)** dan **Bab 4 (Hasil dan Pembahasan)** pada laporan tugas akhir / praktikum *Computer Vision*:

### A. Alur Pipeline Pengolahan Citra
```text
[Citra Asli (BGR/RGB)]
         │
         ▼
[Grayscale uint8 (0-255)]
         │
         ▼
[Pra-Pemrosesan Opsional] ──► Median Filter (noise bintik) / Gaussian Blur / CLAHE
         │
         ▼
[Peta Ambang Lokal T(x,y)] = μ_G(x,y) - C
         │
         ▼
[Adaptive Gaussian Binarization] ──► 255 jika I(x,y) > T(x,y), else 0
         │
         ▼
[Pasca-Pemrosesan Opsional] ──► Morfologi Opening/Closing & Filter Komponen Luas
         │
         ▼
[Citra Biner Bersih Akhir]
```

### B. Formulasi Matematis Inti
1. **Peta Ambang Lokal:**
   $$T(x, y) = \mu_G(x, y) - C$$
   dengan $\mu_G(x, y) = I(x, y) * K_{\text{Gauss}}$ adalah konvolusi citra dengan kernel bobot sebaran normal dua dimensi.
2. **Kernel Gaussian Spasial:**
   $$G(u) = \exp\left(-\frac{u^2}{2\sigma^2}\right), \quad \sigma = 0.3 \times \left(\frac{N - 1}{2} - 1\right) + 0.8$$
   $$K(u, v) = \frac{G(u) \cdot G(v)}{\sum_{i,j} G(i) \cdot G(j)}$$
3. **Keputusan Biner:**
   $$\text{Biner}(x, y) = \begin{cases} 255, & \text{jika } I(x, y) > T(x, y) \\ 0, & \text{lainnya} \end{cases}$$

### C. Alasan Pemilihan Parameter Default
- **Block Size $N = 21$:** Ukuran jendela harus lebih besar dari lebar goresan huruf teks (stroke width) tertebal, namun cukup kecil untuk mengikuti gradien bayangan lokal. Pada dokumen teks beresolusi 500-1000 piksel, goresan huruf memiliki tebal 2-6 piksel. Jendela $21 \times 21$ melingkupi teks sekaligus sampel kertas latar sekitar secara seimbang.
- **Konstanta $C = 5$:** Mengurangi nilai ambang sebesar 5 tingkat keabuan di bawah rata-rata lokal. Nilai ini memberikan toleransi yang cukup terhadap tekstur serat kertas dan fluktuasi intensitas sensor kamera tanpa memotong goresan tinta tipis.
- **Handling Border:** Menggunakan `BORDER_REPLICATE | BORDER_ISOLATED` untuk mencegah tepi dokumen menjadi gelap atau terpotong (*vignetting*).

### D. Perbandingan Terhadap Metode Lain
- **Global Thresholding (T=127):** Gagal total pada dokumen bergradien pencahayaan. Area yang tertutup bayangan menjadi hitam pekat (huruf hilang dalam bayangan), sedangkan area yang terlalu terang membuat huruf tipis memudar.
- **Otsu's Thresholding:** Sangat baik pada citra dengan histogram bimodal yang seragam di seluruh permukaan. Namun, bayangan lokal menciptakan distribusi intensitas multimodal yang menyebabkan ambang batas global salah menempatkan wilayah bayangan sebagai foreground.
- **Adaptive Mean:** Menggunakan bobot kotak seragam ($1 / N^2$). Pada batas tepi yang kontras, timbul efek cincin/artefak kotak (*blocking halo*).
- **Adaptive Gaussian:** Memberikan bobot tertinggi pada piksel terdekat. Menghasilkan kontur goresan huruf yang halus, konsisten, dan sangat tangguh terhadap bayangan lipatan buku diagonal.

---

## 8. Lisensi
Aplikasi ini dikembangkan untuk tujuan edukasi dan riset akademis pada mata kuliah *Computer Vision*. Bebas digunakan dan dimodifikasi untuk keperluan studi dan laporan ilmiah.
