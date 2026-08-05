"""
Pemuatan model Keras (.keras) dengan caching Streamlit, agar model tidak
di-load ulang dari disk setiap kali pengguna berinteraksi dengan UI.
"""

import json
import os

import streamlit as st
import tensorflow as tf

from config import METADATA_PATH, MODEL_OPTIONS


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tf.keras.Model:
    """Load model .keras dari path, di-cache oleh Streamlit (resource cache)
    supaya proses load (yang relatif berat) hanya terjadi sekali per model,
    bukan setiap kali ada interaksi/rerun di UI."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}\n"
            f"Pastikan file .keras hasil training sudah diletakkan di folder models/."
        )
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
    key = "baseline" if model_choice == "CNN Baseline" else "cnn_woa"
    model_metadata = metadata.get(key, {})

    return model, model_metadata
