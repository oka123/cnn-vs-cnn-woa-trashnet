"""
Logika inferensi: menjalankan prediksi model pada citra yang sudah di-preprocess,
serta mengukur waktu inferensi per citra.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES


@dataclass
class PredictionResult:
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict = field(default_factory=dict)  # {class_name: probability}
    inference_time_ms: float = 0.0


def predict_single(model: tf.keras.Model, image_array: np.ndarray, filename: str) -> PredictionResult:
    """
    Jalankan prediksi untuk 1 citra (sudah dalam bentuk array (IMG_SIZE, IMG_SIZE, 3)).
    Waktu inferensi diukur HANYA untuk proses model.predict(), tidak termasuk
    waktu preprocessing, agar angka yang ditampilkan murni performa model.
    """
    batch = np.expand_dims(image_array, axis=0)  # (1, H, W, 3)

    start = time.perf_counter()
    probs = model.predict(batch, verbose=0)[0]  # shape (NUM_CLASSES,)
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return PredictionResult(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        inference_time_ms=elapsed_ms,
    )


from typing import Any
from PIL import Image

def predict_batch(model: Any, image_arrays: list[np.ndarray],
                   filenames: list[str], pil_images: list[Image.Image] = None) -> list[PredictionResult]:
    """
    Jalankan prediksi untuk banyak citra sekaligus dalam satu batch.
    Jika model YOLO, akan menggunakan pil_images.
    Jika Keras, menggunakan image_arrays.
    """
    # Deteksi jika model adalah YOLO dari ultralytics
    if hasattr(model, "__class__") and "YOLO" in model.__class__.__name__:
        start = time.perf_counter()
        yolo_results = model(pil_images, verbose=False)
        elapsed_ms_total = (time.perf_counter() - start) * 1000
        elapsed_ms_per_image = elapsed_ms_total / len(pil_images)
        
        results = []
        for i, filename in enumerate(filenames):
            r = yolo_results[i]
            probs_data = r.probs.data.cpu().numpy()
            
            probabilities = {}
            for cls_idx, prob in enumerate(probs_data):
                cls_name = r.names.get(cls_idx, CLASS_NAMES[cls_idx])
                probabilities[cls_name] = float(prob)
                
            predicted_class = r.names.get(int(r.probs.top1), CLASS_NAMES[int(r.probs.top1)])
            confidence = float(r.probs.top1conf)
            
            results.append(PredictionResult(
                filename=filename,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                inference_time_ms=elapsed_ms_per_image,
            ))
        return results

    # Logika untuk Keras Model
    batch = np.stack(image_arrays, axis=0)  # (N, H, W, 3)

    start = time.perf_counter()
    probs_batch = model.predict(batch, verbose=0)  # (N, NUM_CLASSES)
    elapsed_ms_total = (time.perf_counter() - start) * 1000
    elapsed_ms_per_image = elapsed_ms_total / len(image_arrays)

    results = []
    for i, filename in enumerate(filenames):
        probs = probs_batch[i]
        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])
        probabilities = {CLASS_NAMES[j]: float(probs[j]) for j in range(len(CLASS_NAMES))}

        results.append(PredictionResult(
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=elapsed_ms_per_image,
        ))

    return results
