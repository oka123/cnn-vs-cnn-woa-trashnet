"""
Konfigurasi global aplikasi.
Nilai-nilai di sini HARUS identik dengan konfigurasi saat training
(lihat Bagian 2 - Konfigurasi Global di notebook Kaggle) agar preprocessing
saat inferensi konsisten dengan preprocessing saat training.
"""

import os

# ── Path dasar ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Parameter citra (HARUS sama dengan training) ───────────────────────
IMG_SIZE = 224
IMG_CHANNELS = 3

# Urutan kelas HARUS sama dengan CLASS_NAMES di notebook training
# (urutan ini menentukan index output softmax model)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# Label tampilan yang lebih ramah pengguna (opsional, untuk UI)
CLASS_DISPLAY_NAMES = {
    "cardboard": "Karton (Cardboard)",
    "glass": "Kaca (Glass)",
    "metal": "Logam (Metal)",
    "paper": "Kertas (Paper)",
    "plastic": "Plastik (Plastic)",
    "trash": "Lainnya (Trash)",
}

# Warna representatif tiap kelas (dipakai untuk badge & chart)
CLASS_COLORS = {
    "cardboard": "#B08968",
    "glass": "#4CC9F0",
    "metal": "#8D99AE",
    "paper": "#F4A261",
    "plastic": "#E63946",
    "trash": "#6C757D",
}

# ── Path model & metadata ───────────────────────────────────────────────
MODEL_OPTIONS = {
    "CNN Random Search": {
        "path": os.path.join(MODELS_DIR, "cnn_random_search_final.keras"),
        "description": "CNN dengan hyperparameter hasil pencarian acak (Random Search).",
        "metadata_key": "random_search"
    },
    "CNN-WOA": {
        "path": os.path.join(MODELS_DIR, "cnn_woa_final.keras"),
        "description": "CNN dengan hyperparameter hasil optimasi Whale Optimization "
                        "Algorithm (WOA).",
        "metadata_key": "cnn_woa"
    },
    "EfficientNetB0": {
        "path": os.path.join(MODELS_DIR, "EfficientNetB0_trashnet.keras"),
        "description": "Model klasifikasi menggunakan arsitektur EfficientNetB0 pre-trained.",
        "metadata_key": "efficientnetb0"
    },
    "YOLO26": {
        "path": os.path.join(MODELS_DIR, "yolo_saved_model"),
        "description": "Model klasifikasi TrashNet menggunakan arsitektur YOLOv26 dari Ultralytics.",
        "metadata_key": "yolo"
    }
}

METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ── Batas unggah ─────────────────────────────────────────────────────────
MAX_UPLOAD_FILES = 20
ALLOWED_EXTENSIONS = [
    "jpg", "jpeg", "png", "webp",  # Format umum
    "bmp", "gif", "tiff", "tif",  # Format lossless
    "avif", "heic", "heif",       # Format modern
    "ppm", "pgm", "ico",          # Format lainnya
]
