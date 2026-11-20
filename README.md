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

Struktur repositori yang terlacak pada Git (sesuai aturan `.gitignore`):

```text
AdaptiveLab/
├── .gitignore                  # Konfigurasi pengabaian berkas & folder Git
├── app.py                      # Server Flask, routing API JSON, & auto-discovery dataset
├── requirements.txt            # Dependensi pustaka Python
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
├── sample_images/              # Dataset naskah kuno H-DIBCO 2016 & Ground Truth
│   ├── 1.bmp ... 10.bmp        # Citra dokumen tulisan tangan H-DIBCO 2016
│   └── 1_gt.bmp ... 10_gt.bmp  # Citra acuan Ground Truth biner resmi
└── tests/                      # Rangkaian pengujian otomatis Pytest
    ├── test_threshold.py       # Validasi parameter & uji identik manual vs opencv (< 1%)
    ├── test_modules.py         # Pengujian preprocess, postprocess, metrics, compare
    └── test_api.py             # Pengujian endpoint REST API Flask
```

> **Catatan Berkas yang Diabaikan (*Ignored by `.gitignore`*):**
> - `report_assets/` (direktori aset visual laporan bab 4)
> - `scripts/` (`make_samples.py` & `generate_report_assets.py`)
> - `sample_images/sample*.png` (sampel sintetis bawaan diabaikan, digantikan oleh H-DIBCO 2016)
> - `outputs/` (direktori penyimpanan sementara server untuk unggahan dan ekspor)
> - `.venv/`, `__pycache__/`, `.pytest_cache/` (lingkungan virtual dan cache Python)

---

## 3. Cara Instalasi dan Menjalankan di Windows

Pastikan Python 3.10 atau versi yang lebih baru telah terpasang di sistem Windows Anda.

### Langkah 1: Buat & Aktifkan Lingkungan Virtual (Virtual Environment)
```powershell
py -m venv .venv
.venv\Scripts\activate
```

### Langkah 2: Pasang Dependensi
```powershell
pip install -r requirements.txt
```

### Langkah 3: Jalankan Server Aplikasi
```powershell
py app.py
```

### Langkah 4: Buka Aplikasi di Peramban Web
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

## 5. Penggunaan Dataset H-DIBCO 2016 (Handwritten Document Image Binarization Contest)

Aplikasi ini dioptimalkan dan diuji menggunakan dataset standar kompetisi internasional **H-DIBCO 2016** (*Handwritten Document Image Binarization Contest*):
- **Situs Resmi:** [https://vc.ee.duth.gr/h-dibco2016/](https://vc.ee.duth.gr/h-dibco2016/)
- **Penyelenggara:** Democritus University of Thrace (DUTH) & ICFHR (International Conference on Frontiers in Handwriting Recognition) 2016.

### A. Karakteristik & Tantangan Dataset H-DIBCO 2016
Dataset H-DIBCO 2016 terdiri dari dokumen naskah tulisan tangan kuno bersejarah yang memiliki berbagai degradasi kompleks:
1. **Tinta Tembus (*Bleed-Through / Show-Through*):** Goresan tulisan tinta dari halaman belakang menembus ke halaman depan kertas lapuk.
2. **Noda Kelembapan & Jamur (*Foxing*):** Bercak-bercak cokelat/kekuningan akibat degradasi kimiawi kertas kuno.
3. **Pencahayaan & Bayangan Tidak Merata:** Variasi iluminasi akibat kelengkungan jilidan naskah atau kondisi pemindaian/fotografi arsip.
4. **Intensitas Tinta Pudar (*Faded Ink*):** Goresan pena tipis yang mengalami penurunan kontras terhadap latar kertas.

### B. Rekomendasi Parameter Pengujian untuk Dokumen H-DIBCO 2016
Berdasarkan karakteristik naskah tulisan tangan H-DIBCO 2016:
- **Block Size ($N$):** Gunakan $N = 25$ hingga $35$ (ganjil). Rentang ini ideal untuk melingkupi lebar goresan pena tulisan tangan serta spasi antar-kata.
- **Konstanta $C$:** Disarankan menaikkan nilai $C$ ke rentang **$8$ hingga $15$** (default: 5). Nilai $C$ yang lebih tinggi efektif menekan noise *bleed-through* (tinta tembus) dan noda kertas agar tidak salah terklasifikasi sebagai teks.
- **Pra-pemrosesan:** Aktifkan **Gaussian Blur** ($3 \times 3$, $\sigma = 0.8$) atau **Median Blur** ($3 \times 3$) untuk menghaluskan tekstur serat kertas kuno sebelum binerisasi.
- **Pasca-pemrosesan:** Aktifkan **Hapus Komponen Kecil** (*min area* $8 - 15$ px) untuk membersihkan bintik-bintik tinta tembus terisolasi.

### C. Format Evaluasi Standar & Atribusi Ilmiah
AdaptiveLab menghitung metrik kuantitatif resmi yang selaras dengan standar kompetisi DIBCO:
- **F-Measure (F1-Score):** Rata-rata harmonik Precision dan Recall deteksi goresan teks.
- **PSNR (Peak Signal-to-Noise Ratio):** Tingkat kemiripan visual terhadap Ground Truth biner.
- **Confusion Map:** Memvisualisasikan piksel teks terdeteksi benar (hijau), noise/bleed-through bocor (merah), dan teks terputus (biru).

> **Sitasi Ilmiah untuk Laporan Tugas/Makalah:**  
> I. Pratikakis, K. Zagoris, G. Barlas, and B. Gatos, *"ICFHR2016 Handwritten Document Image Binarization Contest (H-DIBCO 2016)"*, in Proceedings of the 15th International Conference on Frontiers in Handwriting Recognition (ICFHR), pp. 619-623, 2016. URL: [https://vc.ee.duth.gr/h-dibco2016/](https://vc.ee.duth.gr/h-dibco2016/)

---

## 6. Lisensi
Aplikasi ini dikembangkan untuk tujuan edukasi dan riset akademis pada mata kuliah *Computer Vision*. Bebas digunakan dan dimodifikasi untuk keperluan studi dan laporan ilmiah.
