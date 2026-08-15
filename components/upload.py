"""Komponen unggah citra (satu atau banyak) + preview thumbnail."""

import streamlit as st

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_FILES
from utils.preprocessing import load_image_from_bytes


def render_upload_section():
    """Render uploader + preview grid. Return list of (filename, PIL.Image)."""
    st.markdown("### 📤 Unggah Citra")

    tab1, tab2 = st.tabs(["📁 Unggah File", "📷 Kamera"])
    
    all_uploaded_files = []
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa gambar sampah",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Format didukung: {', '.join(ALLOWED_EXTENSIONS)}. Maksimum {MAX_UPLOAD_FILES} file.",
        )
        if uploaded_files:
            all_uploaded_files.extend(uploaded_files)

    with tab2:
        camera_file = st.camera_input("Ambil gambar dari kamera")
        if camera_file is not None:
            # camera_file.name is generic like "camera_image.jpeg", which is fine.
            all_uploaded_files.append(camera_file)

    if not all_uploaded_files:
        st.info("👆 Unggah gambar atau gunakan kamera untuk memulai klasifikasi.", icon="ℹ️")
        return []

    if len(all_uploaded_files) > MAX_UPLOAD_FILES:
        st.warning(
            f"Anda mengunggah {len(all_uploaded_files)} file, tapi maksimum yang diproses "
            f"adalah {MAX_UPLOAD_FILES}. Hanya {MAX_UPLOAD_FILES} file pertama yang dipakai."
        )
        all_uploaded_files = all_uploaded_files[:MAX_UPLOAD_FILES]

    uploaded_files = all_uploaded_files

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
