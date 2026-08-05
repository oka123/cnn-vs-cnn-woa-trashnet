"""Komponen tampilan hasil prediksi: kelas, confidence, tabel & bar chart
probabilitas seluruh kelas, waktu inferensi -- per citra & ringkasan agregat."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CLASS_COLORS, CLASS_DISPLAY_NAMES
from utils.inference import PredictionResult


def _confidence_badge(confidence: float) -> str:
    """Badge warna berdasar tingkat keyakinan prediksi."""
    if confidence >= 0.80:
        return f":green[**Tinggi** ({confidence*100:.1f}%)]"
    elif confidence >= 0.50:
        return f":orange[**Sedang** ({confidence*100:.1f}%)]"
    else:
        return f":red[**Rendah** ({confidence*100:.1f}%)]"


def _render_probability_chart(probabilities: dict, key: str):
    """Bar chart horizontal probabilitas seluruh kelas, diurutkan menurun."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [CLASS_DISPLAY_NAMES.get(c, c) for c, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]
    colors = [CLASS_COLORS.get(c, "#4C72B0") for c, _ in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis_title="Probabilitas (%)",
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=10, r=25, t=10, b=10),
        xaxis_range=[0, 115],
    )
    st.plotly_chart(fig, width="stretch", key=key)


def render_single_result(pil_image, preprocessed_array, result: PredictionResult, index: int):
    """Render 1 kartu hasil prediksi lengkap (gambar asli vs preprocessed, kelas, confidence, tabel, chart, waktu)."""
    with st.container(border=True):
        col_img, col_info = st.columns([1, 2], gap="medium")

        with col_img:
            tab_orig, tab_prep = st.tabs(["🖼️ Citra Asli", "⚙️ Preprocessed"])
            with tab_orig:
                st.image(pil_image, width="stretch", caption="Citra Asli")
            with tab_prep:
                st.image(
                    preprocessed_array,
                    width="stretch",
                    caption="Preprocessed (224×224, [0,1])",
                )

        with col_info:
            display_class = CLASS_DISPLAY_NAMES.get(result.predicted_class, result.predicted_class)
            st.markdown(f"#### 🏷️ {display_class}")
            st.markdown(f"**Tingkat Keyakinan:** {_confidence_badge(result.confidence)}")

            m1, m2 = st.columns(2)
            m1.metric("Confidence Score", f"{result.confidence*100:.2f}%")
            m2.metric("Waktu Inferensi", f"{result.inference_time_ms:.1f} ms")

        st.markdown("**Probabilitas Seluruh Kelas**")
        tab_chart, tab_table = st.tabs(["📊 Bar Chart", "📋 Tabel"])

        with tab_chart:
            _render_probability_chart(result.probabilities, key=f"chart_{index}")

        with tab_table:
            df = pd.DataFrame({
                "Kelas": [CLASS_DISPLAY_NAMES.get(c, c) for c in result.probabilities.keys()],
                "Probabilitas (%)": [v * 100 for v in result.probabilities.values()],
            }).sort_values("Probabilitas (%)", ascending=False).reset_index(drop=True)

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Probabilitas (%)": st.column_config.ProgressColumn(
                        "Probabilitas (%)", min_value=0, max_value=100, format="%.2f%%"
                    )
                },
            )


def render_results_section(images_with_results):
    """
    Render seluruh hasil prediksi: kartu detail per citra + ringkasan agregat.
    images_with_results: list of (filename, PIL.Image, np.ndarray, PredictionResult)
    """
    st.markdown("## 🔍 Hasil Prediksi")

    n = len(images_with_results)
    total_time = sum(r.inference_time_ms for _, _, _, r in images_with_results)
    avg_confidence = sum(r.confidence for _, _, _, r in images_with_results) / n

    # ── Ringkasan agregat ────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Citra Diproses", n)
    s2.metric("Rata-rata Confidence", f"{avg_confidence*100:.2f}%")
    s3.metric("Total Waktu Inferensi", f"{total_time:.1f} ms")

    st.divider()

    # ── Preview Preprocessing Batch & Tabel ringkasan ───────────────
    with st.expander("⚙️ Lihat Hasil Preprocessing Seluruh Citra (224×224)"):
        st.caption("Hasil resize ke 224×224 piksel dan normalisasi [0, 1] sebelum dikirim ke model:")
        cols_prep = st.columns(min(n, 4))
        for idx, (fname, _, prep_arr, _) in enumerate(images_with_results):
            with cols_prep[idx % 4]:
                st.image(prep_arr, caption=f"{fname}\n(224×224×3)", width="stretch")

    # ── Tabel ringkasan agregat (bisa diunduh) ────────────────────────
    summary_df = pd.DataFrame([
        {
            "Nama File": filename,
            "Kelas Prediksi": CLASS_DISPLAY_NAMES.get(result.predicted_class, result.predicted_class),
            "Confidence (%)": round(result.confidence * 100, 2),
            "Waktu Inferensi (ms)": round(result.inference_time_ms, 2),
        }
        for filename, _, _, result in images_with_results
    ])

    with st.expander("📄 Tabel Ringkasan Seluruh Hasil", expanded=(n > 1)):
        st.dataframe(summary_df, width="stretch", hide_index=True)
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Unduh Ringkasan (CSV)", data=csv,
            file_name="hasil_klasifikasi_sampah.csv", mime="text/csv",
        )

    st.divider()

    # ── Kartu detail per citra ─────────────────────────────────────────
    for i, (filename, pil_image, prep_arr, result) in enumerate(images_with_results):
        st.markdown(f"**{i+1}. {filename}**")
        render_single_result(pil_image, prep_arr, result, index=i)
        st.write("")
