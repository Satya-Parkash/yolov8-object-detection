"""
detect_image.py — Image-based object detection using YOLOv8.

Handles:
  • Uploading a JPG / PNG image through Streamlit
  • Running YOLOv8 inference
  • Displaying annotated results side-by-side with the original
  • Showing per-object statistics (counts, confidence)
  • Offering a one-click download of the annotated image
"""

import os
import io
import time
import numpy as np
import streamlit as st
from PIL import Image

from utils.helper  import load_model, ensure_dirs
from utils.drawing import draw_detections, build_summary_table


# ── Constants ────────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"


def run_image_detection(confidence: float = 0.40):
    """
    Streamlit page for image detection.

    Parameters
    ----------
    confidence : float
        Minimum confidence score (0–1) set via the sidebar slider.
    """

    ensure_dirs([UPLOAD_DIR, OUTPUT_DIR])

    st.title("🖼️ Image Object Detection")
    st.markdown("Upload an image and YOLOv8 will detect all objects in it.")
    st.markdown("---")

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Choose an image (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Max recommended size: 10 MB",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image to begin detection.")
        _show_tips()
        return

    # ── Save upload to disk ──────────────────────────────────────────────────
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ── Load & display original image ────────────────────────────────────────
    original_img = Image.open(save_path).convert("RGB")
    orig_np      = np.array(original_img)

    # ── Run detection ────────────────────────────────────────────────────────
    model = load_model()

    with st.spinner("🔍 Running YOLOv8 detection …"):
        start_time = time.time()
        results    = model.predict(
            source=save_path,
            conf=confidence,
            verbose=False,
        )
        elapsed = time.time() - start_time

    # ── Annotate image ───────────────────────────────────────────────────────
    annotated_np  = draw_detections(orig_np.copy(), results, confidence)
    annotated_img = Image.fromarray(annotated_np)

    # ── Save annotated output ────────────────────────────────────────────────
    output_name = f"detected_{uploaded_file.name}"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    annotated_img.save(output_path)

    # ── Metrics row ──────────────────────────────────────────────────────────
    boxes      = results[0].boxes
    num_dets   = len(boxes) if boxes is not None else 0
    avg_conf   = (
        float(boxes.conf.mean()) if (boxes is not None and len(boxes) > 0) else 0.0
    )
    h, w = orig_np.shape[:2]

    st.markdown("### 📊 Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔎 Objects Found", num_dets)
    c2.metric("📈 Avg Confidence", f"{avg_conf:.1%}")
    c3.metric("⏱️ Inference Time", f"{elapsed:.2f}s")
    c4.metric("📐 Resolution",     f"{w}×{h}")

    st.markdown("---")

    # ── Side-by-side display ─────────────────────────────────────────────────
    st.markdown("### 🖼️ Results")
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("**Original Image**")
        st.image(original_img, use_container_width=True)
    with col_det:
        st.markdown("**Detected Objects**")
        st.image(annotated_img, use_container_width=True)

    # ── Per-class breakdown table ────────────────────────────────────────────
    if num_dets > 0:
        st.markdown("---")
        st.markdown("### 📋 Detected Objects Breakdown")
        df = build_summary_table(results)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ No objects detected above the confidence threshold. "
                   "Try lowering the slider in the sidebar.")

    # ── Download button ──────────────────────────────────────────────────────
    st.markdown("---")
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=buf,
        file_name=output_name,
        mime="image/png",
    )


def _show_tips():
    """Display helpful tips when no image is uploaded yet."""
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use a clear, well-lit image for best results
    - Images with multiple objects work great
    - Lower the **confidence threshold** (sidebar) to catch more objects
    - Supported formats: **JPG, JPEG, PNG**
    """)
