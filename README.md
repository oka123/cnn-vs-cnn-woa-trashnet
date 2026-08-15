# ♻️ Klasifikasi Jenis Sampah Menggunakan CNN Random Search (Metode Baseline) dan CNN-WOA

Aplikasi web berbasis **Streamlit** untuk mengklasifikasikan jenis sampah dari citra,
menggunakan dua model CNN yang dapat dipilih pengguna:

- **CNN Random Search** — arsitektur CNN dengan hyperparameter hasil pencarian acak (Random Search).
- **CNN-WOA** — arsitektur CNN dengan hyperparameter hasil optimasi
  _Whale Optimization Algorithm_ (WOA).

Kedua model dilatih pada dataset [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet)
untuk mengenali 6 kelas: **cardboard, glass, metal, paper, plastic, trash**.

---

## ✨ Fitur

- Pilih model klasifikasi (CNN Random Search / CNN-WOA) beserta info hyperparameter & akurasi.
- Unggah **satu atau banyak** citra sekaligus (drag & drop / file picker).
- Preview thumbnail seluruh citra sebelum diproses.
- Tombol **Prediksi** menjalankan inferensi untuk seluruh citra yang diunggah.
- Hasil prediksi menampilkan:
  - Kelas prediksi + badge tingkat keyakinan (Tinggi/Sedang/Rendah)
  - Confidence score (%)
  - Tabel probabilitas seluruh kelas (dengan progress bar visual)
  - Bar chart interaktif probabilitas seluruh kelas
  - Waktu inferensi (ms) per citra
  - Ringkasan agregat (rata-rata confidence, total waktu) + tabel ringkasan yang bisa diunduh (CSV)
- Preprocessing saat inferensi **identik** dengan preprocessing saat training
  (resize 224×224 + normalisasi 0–1), **tanpa augmentasi**.

---

## 📁 Struktur Proyek

```
trashnet_app/
├── app.py                     # Entry point utama Streamlit
├── config.py                  # Konfigurasi global (path, konstanta, warna kelas)
├── requirements.txt
├── README.md
├── models/                    # Letakkan file model & metadata di sini
│   ├── cnn_random_search_final.keras
│   ├── cnn_woa_final.keras
│   ├── model_metadata.json
│   └── PLACE_MODELS_HERE.txt
├── utils/
│   ├── preprocessing.py       # Preprocessing citra (identik dgn training, tanpa augmentasi)
│   ├── model_loader.py        # Load model & metadata (dengan caching)
│   └── inference.py           # Logika prediksi + pengukuran waktu inferensi
├── components/
│   ├── sidebar.py             # Komponen pemilihan model & info pendukung
│   ├── upload.py               # Komponen unggah citra + preview
│   └── results.py              # Komponen tampilan hasil (badge, tabel, chart)
└── assets/
    └── style.css               # Custom CSS (styling & responsivitas)
```

---

## 🚀 Cara Menjalankan

### 1. Clone / salin folder proyek ini

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## ⚙️ Konfigurasi

Semua konstanta penting (ukuran citra, nama kelas, path model) diatur terpusat
di `config.py`. Jika Anda melatih ulang model dengan konfigurasi berbeda
(misal `IMG_SIZE` atau urutan `CLASS_NAMES` berubah), **pastikan nilai di
`config.py` diperbarui agar tetap konsisten dengan model yang di-load** —
inkonsistensi di sini akan menyebabkan prediksi salah tanpa error yang jelas
(mis. urutan kelas tertukar).

---

## 🧩 Catatan

- **Preprocessing**: `utils/preprocessing.py` meniru persis fungsi
  `load_and_preprocess()` pada notebook training — resize via `tf.image.resize`
  lalu normalisasi `/255.0`. Tidak ada augmentasi (flip/rotasi/translasi) yang
  diterapkan saat inferensi, karena augmentasi hanya relevan saat training.
- **Caching**: model di-cache dengan `st.cache_resource` agar tidak di-load
  ulang dari disk setiap kali pengguna berinteraksi dengan UI (mis. mengganti
  pilihan model, mengunggah citra baru).
- **Waktu inferensi**: diukur murni pada pemanggilan `model.predict()`, tidak
  termasuk waktu preprocessing atau rendering UI, agar mencerminkan performa
  model yang sebenarnya.
- **Unggah banyak citra**: diproses sebagai satu batch (`model.predict()`
  dipanggil sekali untuk seluruh citra), lebih efisien dibanding memanggil
  prediksi satu-satu.

---
