"""
detect_webcam.py — Live webcam object detection using YOLOv8.

Handles:
  • Opening the system webcam via OpenCV
  • Running per-frame YOLOv8 inference inside a Streamlit loop
  • Displaying FPS in real time
  • Graceful start / stop controls
  • Snapshot download of any annotated frame
"""

import time
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from utils.helper  import load_model
from utils.drawing import draw_detections


def run_webcam_detection(confidence: float = 0.40):
    """
    Streamlit page for live webcam detection.

    Parameters
    ----------
    confidence : float
        Minimum confidence score set via the sidebar slider.
    """

    st.title("📷 Live Webcam Detection")
    st.markdown(
        "Click **▶ Start Webcam** to begin real-time object detection "
        "using your connected camera."
    )
    st.markdown("---")

    # ── Controls row ─────────────────────────────────────────────────────────
    col_start, col_stop, col_snap = st.columns([1, 1, 2])
    start_btn    = col_start.button("▶ Start Webcam",  type="primary")
    stop_btn     = col_stop.button("⏹ Stop Webcam")
    snapshot_btn = col_snap.button("📸 Take Snapshot")

    st.markdown("---")

    # ── Metrics placeholders ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    fps_display  = m1.empty()
    obj_display  = m2.empty()
    conf_display = m3.empty()
    frame_disp   = m4.empty()

    fps_display.metric("⚡ FPS",         "—")
    obj_display.metric("🔎 Objects",      "—")
    conf_display.metric("📈 Avg Conf",    "—")
    frame_disp.metric("🎞️ Frame #",      "—")

    # ── Video frame slot ─────────────────────────────────────────────────────
    frame_slot    = st.empty()
    snapshot_slot = st.empty()

    # ── Session-state flag — persists across re-runs ─────────────────────────
    if "webcam_running" not in st.session_state:
        st.session_state["webcam_running"] = False
    if "last_frame" not in st.session_state:
        st.session_state["last_frame"] = None

    if start_btn:
        st.session_state["webcam_running"] = True

    if stop_btn:
        st.session_state["webcam_running"] = False

    # ── Snapshot download (uses last captured frame) ──────────────────────────
    if snapshot_btn and st.session_state["last_frame"] is not None:
        snap_img = Image.fromarray(st.session_state["last_frame"])
        buf = io.BytesIO()
        snap_img.save(buf, format="PNG")
        buf.seek(0)
        snapshot_slot.download_button(
            label="⬇️ Download Snapshot",
            data=buf,
            file_name="snapshot.png",
            mime="image/png",
        )
    elif snapshot_btn:
        snapshot_slot.warning("⚠️ Start the webcam first, then take a snapshot.")

    # ── Main capture loop ─────────────────────────────────────────────────────
    if st.session_state["webcam_running"]:
        model = load_model()

        cap = cv2.VideoCapture(0)   # 0 = default webcam; change for external cams
        if not cap.isOpened():
            st.error(
                "❌ Could not open webcam. "
                "Please ensure a webcam is connected and not in use by another app."
            )
            st.session_state["webcam_running"] = False
            return

        frame_count  = 0
        fps_timer    = time.time()
        fps          = 0.0

        # Streamlit re-runs the script on every interaction;
        # we use a while loop here — the Stop button sets the flag to False.
        while st.session_state.get("webcam_running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost webcam feed. Please restart.")
                break

            # BGR → RGB for YOLOv8 & Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # ── Inference ──────────────────────────────────────────────────
            t0      = time.time()
            results = model.predict(
                source=frame_rgb,
                conf=confidence,
                verbose=False,
            )
            inf_time = time.time() - t0

            # ── Annotate ───────────────────────────────────────────────────
            annotated = draw_detections(frame_rgb.copy(), results, confidence)

            # Keep last annotated frame for snapshot
            st.session_state["last_frame"] = annotated

            # ── FPS (rolling over 1-second window) ────────────────────────
            if time.time() - fps_timer >= 1.0:
                fps       = frame_count / (time.time() - fps_timer)
                fps_timer = time.time()
                frame_count = 0

            # ── Metrics update ─────────────────────────────────────────────
            boxes     = results[0].boxes
            num_objs  = len(boxes) if boxes is not None else 0
            avg_conf  = (
                float(boxes.conf.mean()) if (boxes is not None and len(boxes) > 0)
                else 0.0
            )

            fps_display.metric("⚡ FPS",        f"{fps:.1f}")
            obj_display.metric("🔎 Objects",     num_objs)
            conf_display.metric("📈 Avg Conf",   f"{avg_conf:.1%}")
            frame_disp.metric("🎞️ Frame #",      frame_count)

            # ── Display ───────────────────────────────────────────────────
            frame_slot.image(annotated, channels="RGB", use_container_width=True)

        cap.release()
        st.info("📷 Webcam stopped. Click **▶ Start Webcam** to resume.")

    else:
        # Show placeholder when webcam is off
        frame_slot.markdown(
            """
            <div style='
                background:#16213e;
                border:2px dashed #e94560;
                border-radius:12px;
                height:400px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:1.4rem;
                color:#888;
            '>
                📷 Webcam feed will appear here
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Tips ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ensure your webcam is **not in use** by another application (e.g. Zoom, Teams)
    - Use **good lighting** for better detection accuracy
    - Press **⏹ Stop Webcam** when finished to free the camera resource
    - Click **📸 Take Snapshot** while running to save the current frame
    - Increase the **confidence slider** to reduce false positives
    """)
