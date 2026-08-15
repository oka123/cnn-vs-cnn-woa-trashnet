"""
Pemuatan model Keras (.keras) dengan caching Streamlit, agar model tidak
di-load ulang dari disk setiap kali pengguna berinteraksi dengan UI.
"""

import json
import os

import streamlit as st
import tensorflow as tf

from config import METADATA_PATH, MODEL_OPTIONS, IMG_SIZE, IMG_CHANNELS, NUM_CLASSES


def build_efficientnet_model() -> tf.keras.Model:
    """Membangun ulang arsitektur EfficientNetB0 secara manual untuk menghindari 
    bug Keras 3 saat memuat TFOpLambda dari file Keras 2 (.h5)."""
    input_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    inputs = tf.keras.Input(shape=input_shape)
    
    preprocess = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_tensor=preprocess
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0")


from typing import Any

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Any:
    """Load model .keras atau .pt dari path, di-cache oleh Streamlit."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}\n"
            f"Pastikan file model sudah diletakkan di folder models/."
        )
        
    if model_path.endswith(".pt"):
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Model YOLO tidak dapat dimuat di lingkungan ini.\n"
                f"Kemungkinan penyebab: konflik TensorFlow + PyTorch di server cloud.\n"
                f"Detail error: {e}"
            ) from e
        
    if model_path.endswith(".h5") and "EfficientNet" in model_path:
        model = build_efficientnet_model()
        model.load_weights(model_path)
        return model

    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load metadata hyperparameter & test accuracy tiap model (hasil Bagian 13
    notebook training), dipakai untuk menampilkan info pendukung di sidebar."""
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def get_model_and_metadata(model_choice: str):
    """Helper: ambil model (di-cache) + info metadata untuk pilihan model tertentu."""
    model_info = MODEL_OPTIONS[model_choice]
    model = load_model(model_info["path"])

    metadata = load_metadata()
    key = model_info.get("metadata_key", "")
    model_metadata = metadata.get(key, {})

    return model, model_metadata
