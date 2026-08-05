"""
Aplikasi Web Klasifikasi Jenis Sampah — CNN Baseline vs CNN-WOA
================================================================
Entry point utama aplikasi Streamlit. Menghubungkan seluruh komponen
modular (sidebar, upload, inference, results) menjadi satu alur:
    1. Pilih model (sidebar)
    2. Unggah citra (satu/banyak)
    3. Klik tombol "Prediksi"
    4. Tampilkan hasil: kelas, confidence, tabel & chart probabilitas, waktu inferensi
"""

import os

import streamlit as st

from components.results import render_results_section
from components.sidebar import render_sidebar
from components.upload import render_upload_section
from utils.inference import predict_batch
from utils.preprocessing import preprocess_batch

# ── Konfigurasi halaman ───────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Sampah — CNN vs CNN-WOA",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    _inject_custom_css()

    # ── Header ────────────────────────────────────────────────────────
    st.title("♻️ Klasifikasi Jenis Sampah")
    st.markdown(
        "Aplikasi ini membandingkan performa **CNN Baseline** dan **CNN-WOA** "
        "(CNN dengan hyperparameter hasil optimasi *Whale Optimization Algorithm*) "
        "dalam mengklasifikasikan jenis sampah dari citra."
    )
    st.divider()

    # ── Sidebar: pilih model ─────────────────────────────────────────
    model_choice, model, model_metadata = render_sidebar()

    # ── Unggah citra ──────────────────────────────────────────────────
    images = render_upload_section()

    if not images:
        return

    st.divider()

    # ── Tombol prediksi ──────────────────────────────────────────────
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button(
            "🔍 Prediksi", type="primary", width="stretch"
        )
    with col_note:
        st.caption(
            f"Model aktif: **{model_choice}**. Prediksi akan dijalankan untuk "
            f"seluruh {len(images)} citra yang diunggah."
        )

    if not predict_clicked:
        return

    # ── Preprocessing + Inferensi ─────────────────────────────────────
    with st.spinner("Memproses citra & menjalankan prediksi..."):
        try:
            filenames = [name for name, _ in images]
            pil_images = [img for _, img in images]

            image_arrays = preprocess_batch(pil_images)  # (N, 224, 224, 3), tanpa augmentasi
            results = predict_batch(model, list(image_arrays), filenames)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses inferensi: {e}", icon="⚠️")
            return

    st.divider()

    # ── Tampilkan hasil ────────────────────────────────────────────────
    images_with_results = [
        (filenames[i], pil_images[i], image_arrays[i], results[i]) for i in range(len(images))
    ]
    render_results_section(images_with_results)


if __name__ == "__main__":
    main()
