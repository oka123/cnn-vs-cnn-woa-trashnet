"""Komponen unggah citra (satu atau banyak) + preview thumbnail."""

import streamlit as st

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_FILES
from utils.preprocessing import load_image_from_bytes


def render_upload_section():
    """Render uploader + preview grid. Return list of (filename, PIL.Image)."""
    st.markdown("### 📤 Unggah Citra")

    uploaded_files = st.file_uploader(
        "Pilih satu atau beberapa gambar sampah",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"Format didukung: {', '.join(ALLOWED_EXTENSIONS)}. Maksimum {MAX_UPLOAD_FILES} file.",
    )

    if not uploaded_files:
        st.info("👆 Unggah gambar untuk memulai klasifikasi.", icon="ℹ️")
        return []

    if len(uploaded_files) > MAX_UPLOAD_FILES:
        st.warning(
            f"Anda mengunggah {len(uploaded_files)} file, tapi maksimum yang diproses "
            f"adalah {MAX_UPLOAD_FILES}. Hanya {MAX_UPLOAD_FILES} file pertama yang dipakai."
        )
        uploaded_files = uploaded_files[:MAX_UPLOAD_FILES]

    images = []
    for file in uploaded_files:
        try:
            pil_image = load_image_from_bytes(file.getvalue())
            images.append((file.name, pil_image))
        except Exception as e:
            st.error(f"Gagal membaca file **{file.name}**: {e}")

    # ── Preview grid ─────────────────────────────────────────────────
    st.markdown(f"**{len(images)} citra siap diproses:**")
    cols = st.columns(min(len(images), 5))
    for i, (filename, pil_image) in enumerate(images):
        with cols[i % 5]:
            st.image(pil_image, caption=filename, width="stretch")

    return images
