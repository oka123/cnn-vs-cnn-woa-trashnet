"""
Preprocessing citra untuk inferensi.

PENTING: langkah-langkah di sini harus identik dengan fungsi load_and_preprocess()
pada notebook training (Bagian 5.2), yaitu:
    1. Decode citra
    2. Resize ke (IMG_SIZE, IMG_SIZE)
    3. Normalisasi piksel ke rentang [0, 1]
TANPA augmentasi apa pun (augmentasi hanya dipakai saat training, tidak saat inferensi).
"""

import io

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Buka file gambar (bytes) menjadi objek PIL Image, dipaksa ke mode RGB
    (menangani kasus citra grayscale, RGBA/transparan, atau CMYK yang kadang
    diunggah pengguna)."""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocessing 1 citra PIL menjadi array siap-prediksi, MENIRU PERSIS
    langkah load_and_preprocess() saat training:
        - resize ke (IMG_SIZE, IMG_SIZE) memakai tf.image.resize (bilinear, sama
          seperti training)
        - normalisasi ke rentang 0-1 (dibagi 255.0)

    Returns
    -------
    np.ndarray, shape (IMG_SIZE, IMG_SIZE, 3), dtype float32, rentang [0, 1]
    """
    img_array = np.array(pil_image, dtype=np.uint8)  # (H, W, 3), 0-255

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])  # sama seperti training
    # img_tensor = tf.image.resize_with_pad(img_tensor, IMG_SIZE, IMG_SIZE)
    img_tensor = img_tensor / 255.0  # normalisasi 0-1, sama seperti training

    return img_tensor.numpy()


def preprocess_batch(pil_images: list[Image.Image]) -> np.ndarray:
    """Preprocessing sekumpulan citra sekaligus menjadi 1 batch array,
    siap dipakai model.predict(). Shape: (N, IMG_SIZE, IMG_SIZE, 3)."""
    processed = [preprocess_image(img) for img in pil_images]
    return np.stack(processed, axis=0)
