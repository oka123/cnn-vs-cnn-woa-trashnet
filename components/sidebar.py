"""Komponen sidebar: pemilihan model + info pendukung."""

import streamlit as st

from config import CLASS_DISPLAY_NAMES, CLASS_NAMES, MODEL_OPTIONS
from utils.model_loader import get_model_and_metadata


def render_sidebar():
    """Render sidebar dan return (model_choice, model, model_metadata)."""
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan Model")

        model_choice = st.selectbox(
            "Pilih model klasifikasi",
            options=list(MODEL_OPTIONS.keys()),
            help="CNN Random Search menggunakan hyperparameter hasil pencarian acak. "
                 "CNN-WOA menggunakan hyperparameter hasil optimasi Whale Optimization Algorithm.",
        )

        st.caption(MODEL_OPTIONS[model_choice]["description"])

        with st.spinner(f"Memuat model {model_choice}..."):
            try:
                model, model_metadata = get_model_and_metadata(model_choice)
            except RuntimeError as e:
                st.error(
                    f"⚠️ **Model tidak dapat dimuat di lingkungan ini.**\n\n{e}\n\n"
                    "Silahkan pilih model lain (CNN Random Search atau CNN-WOA) atau kunjungi web model YOLO26 terpisah di link https://yolo-trashnet.streamlit.app/",
                    icon="🚫",
                )
                st.stop()

        st.success(f"Model **{model_choice}** siap digunakan.", icon="✅")

        # ── Info pendukung model ──────────────────────────────────────
        if model_metadata:
            st.markdown("### 📊 Info Model")
            test_acc = model_metadata.get("test_accuracy")
            if test_acc is not None:
                delta_val = "+6.84% vs Random Search" if model_choice == "CNN-WOA" else None
                st.metric(
                    "Akurasi pada Data Uji",
                    f"{test_acc * 100:.2f}%",
                    delta=delta_val,
                    delta_color="normal" if delta_val else "off",
                )

            hyperparams = model_metadata.get("hyperparameters", {})
            if hyperparams:
                with st.expander("Detail Hyperparameter"):
                    for key, value in hyperparams.items():
                        st.write(f"**{key}**: {value}")

        st.divider()

        # ── Daftar kelas yang dikenali ────────────────────────────────
        st.markdown("### 🏷️ Kelas yang Dikenali")
        for cls in CLASS_NAMES:
            st.write(f"- {CLASS_DISPLAY_NAMES.get(cls, cls)}")

        st.divider()
        st.caption(
            "Aplikasi ini mengklasifikasikan jenis sampah dari citra menggunakan "
            "CNN yang dilatih pada dataset TrashNet. Preprocessing saat inferensi "
            "(resize 224×224 + normalisasi) identik dengan proses saat training, "
            "tanpa augmentasi."
        )

    return model_choice, model, model_metadata
