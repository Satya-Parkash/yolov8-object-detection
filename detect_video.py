"""
detect_video.py — Video-based object detection using YOLOv8.

Handles:
  • Uploading an MP4 / AVI / MOV video
  • Processing every frame through YOLOv8
  • Displaying a live progress bar while processing
  • Showing a preview of the annotated video inside the app
  • Offering a download of the processed video
"""

import os
import io
import time
import tempfile
import numpy as np
import cv2
import streamlit as st

from utils.helper  import load_model, ensure_dirs
from utils.drawing import draw_detections, build_summary_table


# ── Constants ────────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Hard cap: only process this many frames to keep demo fast.
# Set to None (or a very large number) for full video processing.
MAX_FRAMES = 300


def run_video_detection(confidence: float = 0.40):
    """
    Streamlit page for video detection.

    Parameters
    ----------
    confidence : float
        Minimum confidence score (0–1) set via the sidebar slider.
    """

    ensure_dirs([UPLOAD_DIR, OUTPUT_DIR])

    st.title("🎬 Video Object Detection")
    st.markdown(
        "Upload a video file and YOLOv8 will annotate every frame with "
        "bounding boxes, labels, and confidence scores."
    )
    st.markdown("---")

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Choose a video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help=f"Only the first {MAX_FRAMES} frames will be processed in demo mode.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload a video to begin detection.")
        _show_tips()
        return

    # ── Save upload to a temp file so OpenCV can read it ────────────────────
    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1],
        dir=UPLOAD_DIR,
    )
    tfile.write(uploaded_file.read())
    tfile.close()
    input_path = tfile.name

    # ── Open video & collect metadata ────────────────────────────────────────
    cap      = cv2.VideoCapture(input_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_to_process = min(total_fr, MAX_FRAMES)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎞️ Total Frames",   total_fr)
    c2.metric("⚡ FPS",            f"{fps:.1f}")
    c3.metric("📐 Resolution",     f"{width}×{height}")
    c4.metric("🔄 Processing",     f"{frames_to_process} frames")

    st.markdown("---")

    # ── Output video writer ──────────────────────────────────────────────────
    output_name = f"detected_{os.path.basename(uploaded_file.name)}"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model()

    # ── Progress bar + processing loop ───────────────────────────────────────
    st.markdown("### ⏳ Processing Video …")
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    preview_slot  = st.empty()          # live frame preview

    all_results   = []                  # store results for summary table
    start_time    = time.time()

    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB for YOLOv8 (returns BGR-annotated array)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(
            source=frame_rgb,
            conf=confidence,
            verbose=False,
        )

        # Draw annotations onto a copy of the frame (RGB)
        annotated_rgb = draw_detections(frame_rgb.copy(), results, confidence)

        # Convert back to BGR for the VideoWriter
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        writer.write(annotated_bgr)

        all_results.append(results)

        # ── Update UI every 10 frames (keeps the app responsive) ──────────
        if frame_idx % 10 == 0 or frame_idx == frames_to_process - 1:
            pct = int((frame_idx + 1) / frames_to_process * 100)
            progress_bar.progress(pct)

            elapsed   = time.time() - start_time
            frames_done = frame_idx + 1
            eta       = (elapsed / frames_done) * (frames_to_process - frames_done)
            status_text.markdown(
                f"Frame **{frames_done}/{frames_to_process}** — "
                f"Elapsed: `{elapsed:.1f}s` — ETA: `{eta:.1f}s`"
            )

            # Show the latest annotated frame as a preview
            preview_slot.image(
                annotated_rgb, caption=f"Frame {frames_done}", use_container_width=True
            )

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.success(f"✅ Done! Processed {frames_to_process} frames in {total_time:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Detection Statistics (Last Frame)")
    if all_results:
        df = build_summary_table(all_results[-1])
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No objects detected in the last frame above threshold.")

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if os.path.exists(output_path):
        with open(output_path, "rb") as vf:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=vf,
                file_name=output_name,
                mime="video/mp4",
            )
    else:
        st.error("Output video not found. Please try again.")

    # Cleanup temp upload file
    try:
        os.remove(input_path)
    except OSError:
        pass


def _show_tips():
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Short clips (10–30 seconds) process faster
    - MP4 files are recommended for best compatibility
    - The app processes a **maximum of 300 frames** in demo mode
    - Higher confidence thresholds = fewer, more certain detections
    """)
