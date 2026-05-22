"""
app.py — Main entry point for the YOLOv8 Object Detection Streamlit App.

This file sets up the sidebar navigation and routes each page
to the appropriate detection module.
"""

import streamlit as st
from PIL import Image
import os

# ── Page config must be the VERY FIRST Streamlit call ──────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import page modules ─────────────────────────────────────────────────────
from detect_image   import run_image_detection
from detect_video   import run_video_detection
from detect_webcam  import run_webcam_detection

# ── Custom CSS for a polished dark-friendly look ────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    .css-1d391kg { background-color: #1a1a2e; }

    /* Main metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 10px 15px;
        color: white;
    }

    /* Headings accent */
    h1, h2, h3 { color: #e94560 !important; }

    /* Upload area */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #e94560;
        border-radius: 10px;
        padding: 8px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: bold;
    }
    .stButton>button:hover { opacity: 0.85; }

    /* Footer */
    .footer { text-align:center; color:#888; font-size:0.8rem; margin-top:40px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ──────────────────────────────────────────────────────
def sidebar():
    """Render the sidebar logo, navigation, and confidence slider."""

    st.sidebar.markdown("## 🎯 YOLOv8 Detector")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "📌 Navigation",
        ["🏠 Home", "🖼️ Image Detection", "🎬 Video Detection",
         "📷 Webcam Detection", "ℹ️ About Project"],
    )

    st.sidebar.markdown("---")

    # Global confidence threshold shared across all detection pages
    confidence = st.sidebar.slider(
        "🎚️ Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help="Only detections above this score are shown.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color:#888;font-size:0.78rem;text-align:center'>"
        "Powered by <b>YOLOv8</b> · Ultralytics<br>"
        "Built with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

    return page, confidence


# ── Home page ───────────────────────────────────────────────────────────────
def page_home():
    st.title("🎯 Real-Time Multi-Object Detection")
    st.subheader("Powered by YOLOv8 + OpenCV + Streamlit")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧠 Model",    "YOLOv8n")
    col2.metric("📦 Classes",  "80+")
    col3.metric("⚡ Speed",    "Real-Time")
    col4.metric("🖥️ Backend", "PyTorch")

    st.markdown("---")

    st.markdown("""
    ### 👋 Welcome!
    This application lets you run **state-of-the-art object detection** on:

    | Mode | Description |
    |------|-------------|
    | 🖼️ **Image** | Upload a JPG/PNG and detect objects instantly |
    | 🎬 **Video** | Upload an MP4/AVI and process every frame |
    | 📷 **Webcam** | Live detection from your camera feed |

    ### 🛠️ How It Works
    1. Choose a detection mode from the **sidebar**
    2. Adjust the **confidence threshold** to filter weak detections
    3. View annotated results and **download** the output

    ### 🎯 Detectable Objects (COCO — 80 classes)
    """)

    # Show a few example class chips
    classes = [
        "👤 person", "🚗 car", "🏍️ motorbike", "✈️ aeroplane", "🚌 bus",
        "🐶 dog", "🐱 cat", "🍕 pizza", "💻 laptop", "📱 cell phone",
        "🪑 chair", "🧴 bottle", "📺 tv", "⌚ clock", "🎒 backpack",
    ]
    cols = st.columns(5)
    for i, c in enumerate(classes):
        cols[i % 5].success(c)

    st.markdown("---")
    st.markdown(
        "<div class='footer'>YOLOv8 Object Detection App · "
        "College Project · Built with Python & Streamlit</div>",
        unsafe_allow_html=True,
    )


# ── About page ──────────────────────────────────────────────────────────────
def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 🎓 Project Overview
    **Real-Time Multi-Object Detection Using YOLOv8** is a college-level
    computer-vision project that demonstrates how modern deep-learning models
    can detect and localise multiple objects within images, videos, and live
    webcam feeds — all through an intuitive web interface.

    ---

    ## 🛠️ Technologies Used
    | Technology | Role |
    |------------|------|
    | **YOLOv8** (Ultralytics) | Object detection backbone |
    | **Python 3.9+** | Core programming language |
    | **OpenCV** | Frame capture & drawing utilities |
    | **Streamlit** | Interactive web UI |
    | **Pillow** | Image I/O and format conversion |
    | **NumPy** | Array / pixel-level operations |

    ---

    ## 📐 Architecture
    ```
    User Input (image / video / webcam)
           │
           ▼
    Pre-processing  ──────► YOLOv8 Inference
                                   │
                                   ▼
                       Post-processing (NMS, filtering)
                                   │
                                   ▼
                       Annotated Frame / Image
                                   │
                                   ▼
                         Streamlit Display + Download
    ```

    ---

    ## 📚 Learning Outcomes
    - Understanding YOLO architecture and real-time inference
    - Working with pre-trained deep-learning models
    - Integrating OpenCV with Streamlit for live video streams
    - Building modular, production-ready Python applications
    - Handling file uploads and binary downloads in Streamlit

    ---

    ## 🚀 Future Improvements
    - Custom model fine-tuning on domain-specific datasets
    - Object tracking across video frames (SORT / DeepSORT)
    - REST API exposure via FastAPI
    - GPU acceleration (CUDA) for higher FPS
    - Mobile / edge deployment (TFLite / ONNX)

    ---
    """)

    st.info("Built as a Final-Year College Project · YOLOv8 · Python · Streamlit")


# ── Router ──────────────────────────────────────────────────────────────────
def main():
    page, confidence = sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "🖼️ Image Detection":
        run_image_detection(confidence)
    elif page == "🎬 Video Detection":
        run_video_detection(confidence)
    elif page == "📷 Webcam Detection":
        run_webcam_detection(confidence)
    elif page == "ℹ️ About Project":
        page_about()


if __name__ == "__main__":
    main()
